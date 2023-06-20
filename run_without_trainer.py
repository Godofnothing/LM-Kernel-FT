"""Finetuning the library models for sequence classification on GLUE."""

import logging
import os
import sys
from functools import partial
from dataclasses import dataclass, field
from typing import Optional, Union, List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functorch import make_functional_with_buffers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, set_seed

from src.dataset import FewShotDataset
from src.models import ModelForPromptFinetuning, ClassifierWrapper, resize_token_type_embeddings
from src.processors import num_labels_mapping, output_modes_mapping, bound_mapping
from src.linearization import linearize
from src.common import target_transform
from src.engine import train, create_optimizer_and_scheduler

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )
    # Linearize
    linearize: bool = field(
        default=False,
        metadata={"help": "Whether to use linearize model."}
    )
    linearization_type: str = field(
        default='zero_start',
        metadata={"help": "Linearization type."}
    )
    linearization_step: int = field(
        default=0,
        metadata={"help": "Linearization step."}
    )
    # Linear tuning
    from_linearhead: int = field(
        default=False,
        metadata={"help": "Linear tuning."}
    )
    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )
    l2_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use L2 loss."}
    )
    use_task_word: bool = field(
        default=False,
        metadata={'help': 'uses the task words MLM logit for kernel computation'}
    )
    zero_init_head: bool = field(
        default=False,
        metadata={'help': 'Whether to init lm head to zero'}
    )
    # LoRA arguments: only for BERT-type model
    apply_lora: bool = field(
        default=False,
        metadata={'help': 'use LoRA for finetuning'}
    )
    lora_alpha: int = field(
        default=None,
        metadata={'help': 'initialization scale for one of the low rank matrices in lora'}
    )
    lora_r: int = field(
        default=None,
        metadata={'help': 'inner rank for lora matrices'}
    )

@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )

    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: Optional[str] = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={"help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: List[str] = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )


@dataclass
class TrainingArguments:
    # Training
    num_steps: int = field(
        default=1000,
        metadata={"help": "Number of training steps"}
    )
    train_batch_size: int = field(
        default=1,
        metadata={"help": "Training batch size"}
    )
    eval_batch_size: int = field(
        default=1,
        metadata={"help": "Evaluation batch size"}
    )
    # Evaluation
    eval_steps: str = field(
        default='',
        metadata={"help": "Steps when the model is evaluated"}
    )
    # Save params
    output_dir: str = field(
        default='',
        metadata={"help": "Directory to save checkpoints and results."}
    )
    save_steps: str = field(
        default='',
        metadata={"help": "Steps when the model is saved"}
    )
    save_at_the_end: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )
    # Optimizer
    optimizer: str = field(
        default='adam',
        metadata={'help': 'choose sgd or adam. default is adam'}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "Base learning rate."}
    )
    momentum: float = field(
        default=0.9,
        metadata={"help": "SGD momentum"}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "beta_1 in Adam"}
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "beta_2 in Adam"}
    )
    # Scheduler
    lr_scheduler_type: str = field(
        default='constant',
        metadata={"help": "Learning rate scheduler type"}
    )
    # Whether to reinit weights
    random_model_init: bool = field(
        default=False,
        metadata={'help': 'reinit the model randomly'}
    )
    # Misc
    binary_classification: bool = field(
        default=False,
        metadata={"help": "If num_classes=2, convert two softmax logits to single sigmoid logit"}
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Whether to log metrics"}
    )
    log_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to log to W&B"}
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random generator seed"}
    )

@dataclass
class MyDataCollatorWithPadding:
    """
    Implements padding for LM-BFF inputs.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        mask_pos = []
        standard_features = []
        for item in features:
            standard_item = {}
            for field in ["input_ids", "label", "attention_mask", "token_type_ids"]:
                if getattr(item, field) is not None:
                    standard_item[field] = getattr(item, field)
            standard_features.append(standard_item)
            mask_pos.append(item.mask_pos)

        batch = self.tokenizer.pad(
            standard_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if any(mask_pos):
            batch["mask_pos"] = torch.tensor(mask_pos)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, TrainingArguments))
    # parser = HfArgumentParser((ModelArguments, DynamicTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.apply_lora:
        assert 'roberta' in model_args.model_name_or_path, 'LoRA only implemented for RoBERTa models'

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Init W&B logging
    if training_args.log_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "FeatureLearningForLM"),
            name=os.environ.get("WANDB_NAME", "test"),
            config={**model_args.__dict__, **training_args.__dict__, **data_args.__dict__}
        )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id]
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))
            
    os.makedirs(training_args.output_dir, exist_ok=True)

    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == 'prompt-demo':
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail:
            logger.info("Automatically convert the template to GPT-3's in-context learning.")
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ''
            old_template = old_template.replace('*cls*', '')
            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ''
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * instance_id + sent_id))
                # Replace mask
                sub_template = sub_template.replace("*mask*", "*labelx_{}*".format(instance_id))
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + sub_template # Put context at the end
                else:
                    new_template = sub_template + new_template # Put context at the beginning
            logger.info("| {} => {}".format(data_args.template, new_template))
            data_args.template = new_template
        else:
            logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ''
                    old_template = old_template.replace('*cls*', '')
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ''
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * label_id + sent_id))
                        # Replace mask
                        sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                        new_template = new_template + sub_template
                    logger.info("| {} => {}".format(data_args.template_list[i], new_template))
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ''
                old_template = old_template.replace('*cls*', '')
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ''
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                    # Replace mask
                    sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(data_args.template, new_template))
                data_args.template = new_template

    # Create config
    config_kwargs = {'apply_lora': model_args.apply_lora,
                     'lora_alpha': model_args.lora_alpha,
                     'lora_r': model_args.lora_r}
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        **config_kwargs
    )

    config.zero_init_head = model_args.zero_init_head

    if 'prompt' in model_args.few_shot_type:
        model_fn = ModelForPromptFinetuning
    elif model_args.few_shot_type == 'finetune':
        if model_args.from_linearhead:
            model_fn = ModelForPromptFinetuning
        else:
            model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )

    # Get our special datasets.
    train_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type))
    )
    eval_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type))
    )
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type))
    )

    set_seed(training_args.seed)

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    if training_args.random_model_init:
        model.init_weights() # reinit weights to random
    # zero head
    if config.zero_init_head:
        if isinstance(model, ModelForPromptFinetuning):
            if config.model_type == "roberta":
                model.lm_head.decoder.weight.data.zero_()
                model.lm_head.decoder.bias.data.zero_()
            elif config.model_type == "bert":
                model.cls.decoder.weight.data.zero_()
                model.cls.decoder.bias.data.zero_()
            model.classifier.weight.data.zero_()
            model.classifier.bias.data.zero_()
        else:
            model.classifier.out_proj.weight.data.zero_()
            model.classifier.out_proj.bias.data.zero_()

    # remove dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    # wrap model
    if model_args.few_shot_type == 'finetune':
        model = ClassifierWrapper(model)

    model = model.to(device)

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

    # Pass dataset and argument information to the model
    if train_dataset.label_word_list is not None:
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().to(device)
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    if model_args.apply_lora:
        for name, param in model.named_parameters():
            if name.startswith('roberta') and "lora" not in name:
                param.requires_grad_(False)

    if model_args.linearize:
        functional_model, params, buffers = make_functional_with_buffers(model)

        def _func(params, *args, **kwargs): 
            return functional_model(params, buffers, *args, **kwargs)
        
        # linearization 
        params_copy = []
        for param in params:
            params_copy.append(param.detach().clone().requires_grad_(param.requires_grad))
        params_copy = tuple(params_copy)

        model = linearize(_func, params_copy, start=model_args.linearization_type)
    else:
        params = model.parameters()
    
    # create dataloaders
    from torch.utils.data import DataLoader

    data_collator = MyDataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_args.train_batch_size, 
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator,
        num_workers=int(os.environ.get("OMP_NUM_THREADS", 0)),
        drop_last=True
    )

    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=training_args.eval_batch_size, 
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator,
        num_workers=int(os.environ.get("OMP_NUM_THREADS", 0))
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=training_args.eval_batch_size, 
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator,
        num_workers=int(os.environ.get("OMP_NUM_THREADS", 0))
    )

    # create optimizer
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        training_args, 
        params, 
        training_args.num_steps
    )

    # get loss_fn
    if model_args.l2_loss:
        loss_fn = F.mse_loss
        target_transform_fn = partial(target_transform, num_classes=num_labels)
    else:
        loss_fn = F.cross_entropy
        target_transform_fn = None

    # parse eval steps
    eval_steps = []
    if ',' in training_args.eval_steps:
        eval_steps = [int(x) for x in training_args.eval_steps.split(',')]

    # parse save steps
    save_steps = []
    if ',' in training_args.save_steps:
        save_steps = [int(x) for x in training_args.save_steps.split(',')]

    train(
        n_steps=training_args.num_steps,
        model=model,
        params=params,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss_fn=loss_fn,
        target_transform_fn=target_transform_fn,
        eval_steps=eval_steps,
        save_steps=save_steps,
        log_wandb=training_args.log_wandb,
        verbose=training_args.verbose
    )

    print('Foo!')


if __name__ == "__main__":
    main()
