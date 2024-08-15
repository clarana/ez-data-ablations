# data (changes for other dataset classes)
export REMOTE=/net/nfs/allennlp/ananyaj/c4/
export LOCAL=/data/ananyaj/tmp

# checkpoint
# save every 20B tokens
export CHECKPOINT_FOLDER=/data/ananyaj/checkpoints
export SAVE_INTERVAL=10000ba
export NUM_CHECKPOINTS_TO_KEEP=-1

# system
# IMP: batch_size needs to change if num GPUs change
# micro_batch_size is per GPU, that remains constant
# batch_size * gpus = 1024 (1024 sequence each with len 2048 = ~2M tokens/batch)
# micro_batch_size is number of sequences that fit per GPU at once (rest is grad accumulation)
export GPUS=8
export BATCH_SIZE=128
export MICRO_BATCH_SIZE=8
export NUM_WORKERS=8
export PREFETCH_FACTOR=16

# optim
# LR remains same for AdamW, WD changes to 0.1
export OPTIM=lionw
export LR=2e-4
export WEIGHT_DECAY=1e-4

# training
# 160k steps with batch size 2M/batch is 320B tokens total
# warmup is 4B tokens
# cosine decay LR until end of training or at least ~300B tokens
export MAX_DURATION=160000ba
export WARMUP=2000ba
export DECAY_UNTIL=160000ba

# eval
# eval every 2B tokens training tokens
export EVAL_INTERVAL=1000ba
export EVAL_SUBSET_NUM_BATCHES=100
export EVAL_BATCH_SIZE=8

export MODEL_NAME=customgpt-1.1b
export TOKENIZER=gpt2

composer -n $GPUS --master_port 1234 pretrain.py \
    --exp_name $MODEL_NAME-pretrain \
    --model_name $MODEL_NAME \
    --tokenizer_name $TOKENIZER \
    --remote $REMOTE \
    --batch_size $BATCH_SIZE \
    --device_train_microbatch_size $MICRO_BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --optim $OPTIM \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --max_duration $MAX_DURATION \
    --t_warmup $WARMUP \
    --t_max $DECAY_UNTIL \
    --save_interval $SAVE_INTERVAL \
    --num_checkpoints_to_keep $NUM_CHECKPOINTS_TO_KEEP \
    --eval_interval $EVAL_INTERVAL \
    --eval_subset_num_batches $EVAL_SUBSET_NUM_BATCHES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --local $LOCAL \
    --prefetch_factor $PREFETCH_FACTOR \
    --checkpoint_folder $CHECKPOINT_FOLDER
