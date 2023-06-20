#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

TASK="MNLI"           # see all the options in the "cases" below
SEED=13                # random seed and also data seed, by default the data split seeds are {13, 21, 42, 87, 100}
K=16                   # choose from {16, 64, 512} by default
MODEL="roberta-base"   # pick a RoBERTa or BERT model
TYPE="prompt"          # fine-tuning setting, choose from "finetune" and "prompt"
TAG="kernel-prompting" # set a tag to distinguish and aggregate runs in the log
# TYPE="finetune"         # fine-tuning setting, choose from "finetune" and "prompt"
# TAG="kernel-finetuning" # set a tag to distinguish and aggregate runs in the log

export WANDB_NAME="NN-$TASK-$MODEL-$TYPE-$TAG-$K-$SEED"
export WANDB_PROJECT="LM-Kernel-FT"

case $TASK in
    SST-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{'0':'terrible','1':'great'}"
        ;;
    QQP)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    QNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        ;;
    MNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
    SNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    trec)
        TEMPLATE="*cls**mask*:*+sent_0**sep+*"
        MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
        TASK_EXTRA="--first_sent_limit 110"
        ;;
    mr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    cr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    mpqa)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110"
        ;;
    CoLA)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{'0':'incorrect','1':'correct'}"
        ;;
    subj)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{0:'subjective',1:'objective'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    MRPC)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    RTE)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
    sst-5)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
        ;;
    ag_news)
        TEMPLATE=*cls**sent_0*_This_article_is_about*mask*_news.*sep+*
        MAPPING="{1:'world',2:'sports',3:'business',4:'tech'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
esac

python run_without_trainer.py \
    --model_name_or_path=$MODEL \
    --num_steps=1000 \
    --eval_steps=0,2,5,10,20,50,100,200,500,1000 \
    --few_shot_type=$TYPE \
    --task_name=$TASK \
    --template=$TEMPLATE \
    --mapping=$MAPPING \
    --data_dir=data/k-shot-1k-test/$TASK/$K-$SEED \
    --output_dir=result/$TASK-$MODEL-$TYPE-$TAG/$K-$SEED \
    --num_k=$K \
    --verbose \
    --optimizer=sgd \
    --zero_init_head \
    --learning_rate=3.9e-6 \
    --momentum=0.9 \
    --eval_batch_size=16 \
    --train_batch_size=16 \
    --max_seq_length=128 \
    --seed=$SEED \
    --l2_loss \
    --verbose \
    --log_wandb
