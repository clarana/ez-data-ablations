export PARTITION=$1
export SUB_IND=$2
export NUM_SUBSETS=$3

export SOURCE=s2orc-recombined

# data (changes for other dataset classes)

export EXPT_BASE=subinds
export SOURCE=s2orc-recombined

export EXPT=${EXPT_BASE}-part-$SOURCE-${PARTITION//[, ]/}-i${SUB_IND}_of${NUM_SUBSETS}

#export EXPT_BASE=chunksize
#export EXPT=${EXPT_BASE}-part-$SOURCE-${PARTITION//[, ]/}

export PARTITION_INFO=/net/nfs.cirrascale/allennlp/claran/s2orc_partition_idx-9-27.json
export REMOTE=/net/nfs/allennlp/ananyaj/c4/
# export LOCAL=/data/claran/xppt-sources-branched/${SOURCE}/${PARTITION}/
# todo: change to nfs
export LOCAL=/net/nfs.cirrascale/allennlp/claran/xppt-sources-branched/${SOURCE}/${PARTITION}/

export LOCAL_PARENT=/net/nfs.cirrascale/allennlp/claran/xppt-sources-branched/${SOURCE}/
#export LOCAL_EVAL=/data/claran/fake-eval/
export LOCAL_EVAL=/net/nfs.cirrascale/allennlp/claran/eval-perplexity/
# oops not really local but...
export EVAL_GENERAL_PARTS=/net/nfs.cirrascale/allennlp/claran/xppt-sources-final/s2orc/


# checkpoint
# save every 20B tokens
export CHECKPOINT_FOLDER=/data/claran/branch-xppt-${EXPT_BASE}/${SOURCE}/${PARTITION}-i${SUB_IND}
#-many-evals
export SAVE_INTERVAL=20ba
#4000ba
export NUM_CHECKPOINTS_TO_KEEP=1

# system
# IMP: batch_size needs to change if num GPUs change
# micro_batch_size is per GPU, that remains constant
# batch_size * gpus = 1024 (1024 sequence each with len 2048 = ~2M tokens/batch)
# micro_batch_size is number of sequences that fit per GPU at once (rest is grad accumulation)
export GPUS=1
#1
#2

export BATCH_SIZE=1024
#128 # for 8 gpus, but single gpu needs to be 1024
export MICRO_BATCH_SIZE=16
export NUM_WORKERS=2
#4

export PREFETCH_FACTOR=16

# optim
# LR remains same for AdamW, WD changes to 0.1
export OPTIM=adamw
#lionw
export LR=6e-5
export WEIGHT_DECAY=0.1
#1e-4

# training
# 160k steps with batch size 2M/batch is 320B tokens total
# warmup is 800M tokens
# cosine decay LR until end of training or at least ~300B tokens
# export MAX_DURATION=$((20062+${SUB_IND}/4))ba
MAX_DURATION=$((20000+500/${NUM_SUBSETS} + ${SUB_IND}/(${NUM_SUBSETS} - 500 % ${NUM_SUBSETS})))ba
#20500ba
# this is ~1B tokens
#20000ba
export WARMUP=0ba
#400ba
export DECAY_UNTIL=20000ba
#20000ba
export SEED=314159265
# export SEED=271828182

# eval
# eval every 2B tokens training tokens
export EVAL_INTERVAL=100ba
#10ba
#1000ba
export EVAL_SUBSET_NUM_BATCHES=100
export EVAL_BATCH_SIZE=8

export MODEL_PATH=/net/nfs.cirrascale/allennlp/claran/seed-xppt/customgpt-110m-tokens-41943040000-rank-0
# MODEL_PATH=/data/claran/seed-xppt/customgpt-110m-seed-pretrain/customgpt-110m-tokens-41943040000-rank-0
export MODEL_NAME=customgpt-110m
export TOKENIZER=allenai/eleuther-ai-gpt-neox-20b-pii-special
export TOKENIZERS_PARALLELISM=false
#     --partition_info $PARTITION_INFO \
#    --exp_name "${EXPT//[, ]/}" \
#     --eval_perplexity_subsets > >(tee -a "${EXPT//[, ]/}-stdout.log") 2> >(tee -a "${EXPT//[, ]/}-stderr.log" >&2)

# todo: copy model over to checkpoint folder first?

#export HF_DATASETS_CACHE=/net/nfs.cirrascale/allennlp/claran/hf_cache/
#export HF_CACHE=/net/nfs.cirrascale/allennlp/claran/hf_cache

#NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL
composer -n $GPUS --stdout stdout_{rank}.log --stderr stderr_{rank}.log --master_port 1234 pretrain-branch.py \
    --exp_name $EXPT \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
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
    --local_eval $LOCAL_EVAL \
    --local_parts_parent $LOCAL_PARENT \
    --local $LOCAL \
    --prefetch_factor $PREFETCH_FACTOR \
    --checkpoint_folder $CHECKPOINT_FOLDER \
    --seed $SEED \
    --n_gpu $GPUS \
    --eval_ood_partitions \
    --eval_subset_num_batches $EVAL_SUBSET_NUM_BATCHES \
    --resume_training \
    --ood_eval_only \
    --skip_downstream \
    --subset_ind $SUB_IND \
    --num_subsets $NUM_SUBSETS \
    --eval_general_partitions $EVAL_GENERAL_PARTS \
    --eval_perplexity_subsets > >(tee -a "/data/claran/logs/${EXPT}-stdout.log") 2> >(tee -a "/data/claran/logs/${EXPT}-stderr.log" >&2)
    
# cp "$CHECKPOINT_FOLDER/$EXPT-metrics.json" /results/metrics.json

# export checkpt=$(find $CHECKPOINT_FOLDER | grep customgpt-110m-tokens)
# echo "$checkpt"

# if [ $(echo "$checkpt" | wc -l) -gt 1 ]; then 
#     last_checkpt=$(echo "$checkpt" | awk -F- '{split($0, b); print b[length(b)-2]}' | sort -n | tail -n 1);
#     checkpt=$(echo "$checkpt" | grep $last_checkpt);
# fi

# python xp-pt-as-olmo/convert_checkpoint.py --checkpoint $checkpt --config xp-pt-as-olmo/configs/customgpt-110m.yaml --output_dir /net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-${PARTITION//[, ]/}-v2
# cp /net/nfs.cirrascale/allennlp/claran/catwalk-models/seed-xppt/*.json /net/nfs.cirrascale/allennlp/claran/catwalk-models/branched-train-${PARTITION//[, ]/}-v2
