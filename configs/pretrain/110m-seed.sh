export GPUS=$1
export NUM_EPOCH=$2

# data (changes for other dataset classes)
export REMOTE=/net/nfs/allennlp/ananyaj/c4/
export LOCAL=/net/nfs.cirrascale/allennlp/claran/seed-xppt-data
#export LOCAL_EVAL=/data/claran/fake-eval/
export LOCAL_EVAL=/net/nfs.cirrascale/allennlp/claran/eval-perplexity/
# oops not really local but...

# checkpoint
# save every 20B tokens
export EXPT_NAME=${NUM_EPOCH}epoch-decon-paloma
export CHECKPOINT_FOLDER=/data/claran/seed-xppt-${EXPT_NAME}
#-many-evals
export SAVE_INTERVAL=20ba
#4000ba
export NUM_CHECKPOINTS_TO_KEEP=-1

# system
# IMP: batch_size needs to change if num GPUs change
# micro_batch_size is per GPU, that remains constant
# batch_size * gpus = 1024 (1024 sequence each with len 2048 = ~2M tokens/batch)
# micro_batch_size is number of sequences that fit per GPU at once (rest is grad accumulation)
 
export BATCH_SIZE=$((1024/${GPUS}))
export MICRO_BATCH_SIZE=16
export NUM_WORKERS=2

export PREFETCH_FACTOR=16

# optim
# LR remains same for AdamW, WD changes to 0.1
export OPTIM=adamw
#lionw
export LR=6e-4
export WEIGHT_DECAY=0.1
#1e-4

# training
# 160k steps with batch size 2M/batch is 320B tokens total
# warmup is 800M tokens
# cosine decay LR until end of training or at least ~300B tokens
export MAX_DURATION=$((${NUM_EPOCH}*4000))ba
export WARMUP=400ba
export DECAY_UNTIL=$MAX_DURATION
export SEED=15213

# eval
# eval every 2B tokens training tokens
export EVAL_INTERVAL=1000ba
#10ba
#1000ba
export EVAL_SUBSET_NUM_BATCHES=100
export EVAL_BATCH_SIZE=8

export MODEL_NAME=customgpt-110m
export TOKENIZER=allenai/gpt-neox-olmo-dolma-v1_5
#allenai/eleuther-ai-gpt-neox-20b-pii-special
export TOKENIZERS_PARALLELISM=false

#NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL
composer -n $GPUS --stdout stdout_{rank}.log --stderr stderr_{rank}.log --master_port 1234 pretrain-seed.py \
    --exp_name $EXPT_NAME-$MODEL_NAME-seed-pretrain \
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
    --eval_batch_size $EVAL_BATCH_SIZE \
    --local $LOCAL \
    --local_eval $LOCAL_EVAL \
    --prefetch_factor $PREFETCH_FACTOR \
    --checkpoint_folder $CHECKPOINT_FOLDER \
    --seed $SEED \
    --n_gpu $GPUS \
    --eval_subset_num_batches $EVAL_SUBSET_NUM_BATCHES \
    --eval_perplexity_subsets > >(tee -a /data/claran/logs/seed-decon-${NUM_EPOCH}epoch-stdout.log) 2> >(tee -a /data/claran/logs/seed-decon-${NUM_EPOCH}epoch-stderr.log >&2)
#    --skip_downstream \