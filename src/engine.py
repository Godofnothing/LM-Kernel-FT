import os
import math
import torch
import wandb
import torch.nn as nn

from tqdm import tqdm
from typing import Optional
from torch.optim import Optimizer, SGD, AdamW
from timm.utils import AverageMeter
from torch.utils.data import DataLoader

from transformers.optimization import get_linear_schedule_with_warmup

from  .metrics import accuracy


__all__ = [
    'target_transform',
    'create_optimizer_and_scheduler',
    'evaluate',
    'train_step',
    'train'
]


def train_step(
    model, 
    batch,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    target_transform_fn = None,
    do_update = True,
):
    # assuming the model resides on a single device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # move tensors to device
    batch = {k: v.to(device) for k, v in batch.items()}
    targets = batch.pop('labels')

    if target_transform_fn is not None:
        targets = target_transform_fn(targets)
    # get model output
    outputs = torch.stack(model(**batch)).squeeze(-1)
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    if outputs.ndim > targets.ndim:
        outputs = outputs.squeeze(0) 
    # compute loss
    loss = loss_fn(outputs, targets)
    # make gradient step  
    if do_update:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

    return loss.item()


@torch.no_grad()
def evaluate(
    model, 
    loader: DataLoader,
    loss_fn: nn.Module,
    target_transform_fn = None,
):
    acc_m  = AverageMeter()
    loss_m = AverageMeter()
    # assuming the model resides on a single device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch in loader:
        # move tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = batch.pop('labels')

        if target_transform_fn is not None:
            targets = target_transform_fn(targets)
        # get model output
        outputs = torch.stack(model(**batch)).squeeze(-1)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        if outputs.ndim > targets.ndim:
            outputs = outputs.squeeze(0) 
        # compute loss
        loss = loss_fn(outputs, targets)
        acc = accuracy(outputs, targets)
        # update stats
        loss_m.update(loss.item(), len(targets))
        acc_m.update(acc.item(), len(targets))

    return loss_m.avg, acc_m.avg


def create_optimizer_and_scheduler(args, params, num_training_steps: int):
    """
    Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
    are fixed and only the top layers are further fine-tuned.
    """
    if args.optimizer == 'adam':
        optimizer = AdamW(
            params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
    elif args.optimizer == 'sgd':
        optimizer = SGD(
            params,
            lr=args.learning_rate,
            momentum=args.momentum
        )
    else:
        raise NotImplementedError
    
    if args.lr_scheduler_type == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=args.warmup_steps, 
            num_training_steps=num_training_steps
        )
    else:
        lr_scheduler = None

    return optimizer, lr_scheduler


def train(
    n_steps: int,
    model, 
    train_loader: DataLoader, 
    test_loader: DataLoader,
    optimizer: Optimizer, 
    loss_fn: nn.Module,
    params = None, 
    target_transform_fn = None,
    eval_steps: list = [],
    save_steps: list = [],
    save_at_the_end: bool = False,
    output_dir: Optional[str] = None,
    log_wandb: bool = False,
    scheduler = None,
    last_step: int = 0,
    lin_step: int = 0, 
    lin_regime: str='zero_start',
    patience: int = None,
    mov_avg_gamma: float = 0.99,
    loss_rel_tol: float = 0.0,
    verbose: bool = False,
):
    if not isinstance(model, nn.Module):
        assert params is not None
        _model = lambda *args, **kwargs: model(params, *args, **kwargs)
        is_functional = True
    else:
        _model = model
        is_functional = False
        
    summary = {
        'train/loss': [],
        'train/loss_mav': [], 
        'val/loss': [], 
        'val/acc': [], 
        'diverged': False,
        'step': []
    }
    # create loss_tracker
    train_loss_m = AverageMeter()

    step = last_step
    min_train_loss = math.inf
    # create dataloader iterator
    train_iter = iter(train_loader) 
    train_loss = None

    for step in tqdm(range(last_step, n_steps+1)):
        do_update = False if (step < lin_step and lin_regime == 'ref_cont') else True 

        # save model state
        if step in save_steps:
            assert not is_functional
            os.makedirs(f'{output_dir}/step={step}', exist_ok=True)
            torch.save(
                {
                    'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'step': step
                }, 
                f'{output_dir}/step={step}/checkpoint.pth'
            )

        # evaluate model
        if ((step in eval_steps) and do_update) or (step == last_step):
            if test_loader is not None:
                val_loss, val_acc = evaluate(_model, test_loader, loss_fn, target_transform_fn)
            else:
                val_loss, val_acc = 0, 0

        # train step
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_batch = next(train_iter)
  
        train_loss = train_step(
            _model, train_batch,  optimizer, loss_fn, target_transform_fn, do_update
        )
        train_loss_m.update(train_loss)

        # update mov average of loss
        if step == last_step:
            mov_train_loss = train_loss
        else:
            mov_train_loss = (mov_avg_gamma * mov_train_loss + (1 - mov_avg_gamma) * train_loss)
            summary['train/loss_mav'].append(mov_train_loss)

        if patience is not None:
            if mov_train_loss < min_train_loss:
                min_train_loss = mov_train_loss
                countdown = patience
            elif mov_train_loss > min_train_loss * (1 + loss_rel_tol):
                countdown -= 1
            if countdown <= 0 or math.isnan(mov_train_loss):
                summary['diverged'] = True
                print(f"Training diverged on step {step}.")
                break

        # logging
        if step in eval_steps:
            # update summary
            summary['train/loss'].append(train_loss)
            summary['val/loss'].append(val_loss)
            summary['val/acc'].append(val_acc)
            summary['step'].append(step)
            # log
            if verbose:
                print('-' * 10)
                print(f'Step {step}')
                print(f"{'Train':>5} Loss: {train_loss_m.avg:.4f}")
                print(f"{'Val':>5} Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            if log_wandb:
                wandb.log({'train/loss': train_loss_m.avg, 'val/loss': val_loss, 'val/acc': val_acc}, step=step)
            # reset loss meter
            train_loss_m = AverageMeter()

        if scheduler:
            scheduler.step()

    # saving model and optimizer
    if save_at_the_end:
        if is_functional:
            torch.save(
                {
                    'lin_model_params': params, 
                    'optimizer': optimizer.state_dict(),
                    'step': step
                }, 
                f'{output_dir}/step={step}/checkpoint_linearized.pth'
            )
        else:
            torch.save(
                {
                    'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'step': step
                }, 
                f'{output_dir}/step={step}/checkpoint.pth'
            )

    return summary