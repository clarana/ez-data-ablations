export FOS=ALL
export MODEL_SZ=$1

wandb login --relogin $WANDB_KEY

# data (changes for other dataset classes)
export EXPT_BASE=seed40bil-seq-fos-upsamp-decon-$MODEL_SZ
export SOURCE=s2orc-recombined
export EXPT=${EXPT_BASE}-part-$SOURCE-${FOS//[, ]/}

export PARTITION_INFO=/net/nfs.cirrascale/allennlp/claran/s2orc_partition_idx-9-27.json
export REMOTE=/net/nfs/allennlp/ananyaj/c4/

export LOCAL_PARENT=/net/nfs.cirrascale/allennlp/claran/xppt-sources-branched/${SOURCE}/
export LOCAL_EVAL=/net/nfs.cirrascale/allennlp/claran/eval-perplexity/
export EVAL_GENERAL_PARTS=/net/nfs.cirrascale/allennlp/claran/xppt-sources-final/s2orc/


# checkpoint
# save every 20B tokens
export CHECKPOINT_FOLDER=/results/xppt-${EXPT_BASE}/${SOURCE}/${FOS}
export SAVE_INTERVAL=250ba
export NUM_CHECKPOINTS_TO_KEEP=1

# system
# IMP: batch_size needs to change if num GPUs change
# micro_batch_size is per GPU, that remains constant
# batch_size * gpus = 1024 (1024 sequence each with len 2048 = ~2M tokens/batch)
# micro_batch_size is number of sequences that fit per GPU at once (rest is grad accumulation)
export GPUS=8

export BATCH_SIZE=128
#128 # for 8 gpus, but single gpu needs to be 1024
export MICRO_BATCH_SIZE=16
export NUM_WORKERS=2

export PREFETCH_FACTOR=16

# optim
# LR remains same for AdamW, WD changes to 0.1
export OPTIM=adamw

export LR=6e-5
export WEIGHT_DECAY=0.1


# training
# 160k steps with batch size 2M/batch is 320B tokens total
# warmup is 800M tokens
# cosine decay LR until end of training or at least ~300B tokens
export MAX_DURATION=84000ba
#500ba
# this is ~1B tokens
export WARMUP=0ba
export DECAY_UNTIL=20000ba
export SEED=314159265

# eval
# eval every 2B tokens training tokens
export EVAL_INTERVAL=2000ba
#10ba
#1000ba
export EVAL_SUBSET_NUM_BATCHES=100
export EVAL_BATCH_SIZE=8

export MODEL_PATH=/net/nfs.cirrascale/allennlp/claran/seed-xppt-decon-${MODEL_SZ}/customgpt-${MODEL_SZ}-tokens-41943040000-rank-0
# MODEL_PATH=/data/claran/seed-xppt/customgpt-110m-seed-pretrain/customgpt-110m-tokens-41943040000-rank-0
export MODEL_NAME=customgpt-${MODEL_SZ}
export TOKENIZER=allenai/gpt-neox-olmo-dolma-v1_5
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/net/nfs.cirrascale/allennlp/claran/hf_cache

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
    --partition_include "," \
    --upsample \
    --prefetch_factor $PREFETCH_FACTOR \
    --checkpoint_folder $CHECKPOINT_FOLDER \
    --seed $SEED \
    --n_gpu $GPUS \
    --eval_ood_partitions \
    --eval_subset_num_batches $EVAL_SUBSET_NUM_BATCHES \
    --resume_training \
    --ood_eval_only \
    --eval_general_partitions $EVAL_GENERAL_PARTS \
    --eval_perplexity_subsets > >(tee -a "/results/${EXPT}-stdout.log") 2> >(tee -a "/results/${EXPT}-stderr.log" >&2)

cp "$CHECKPOINT_FOLDER/$EXPT-metrics.json" /results/metrics.json

export checkpt=$(find $CHECKPOINT_FOLDER | grep customgpt-${MODEL_SZ}-tokens)
echo "$checkpt"

if [ $(echo "$checkpt" | wc -l) -gt 1 ]; then 
    last_checkpt=$(echo "$checkpt" | awk -F- '{split($0, b); print b[length(b)-2]}' | sort -n | tail -n 1);
    checkpt=$(echo "$checkpt" | grep $last_checkpt);
fi

#    last_checkpt=$(echo "$checkpt" | awk '{split($0, a); print a[1]}' | awk -F- '{split($0, b); print b[length(b)-2]}' | sort -n | tail -n 1)
# export checkpt=$(python -c "print(\"$checkpt_line\".split()[0].strip())")

python xp-pt-as-olmo/convert_checkpoint.py --checkpoint $checkpt --config xp-pt-as-olmo/configs/customgpt-${MODEL_SZ}.yaml --output_dir /net/nfs.cirrascale/allennlp/claran/catwalk-models/seq-fos-decon-$MODEL_SZ/seq-fos-$FOS-upsamp
cp /net/nfs.cirrascale/allennlp/claran/catwalk-models/seed-decon-5epoch-${MODEL_SZ}/*.json /net/nfs.cirrascale/allennlp/claran/catwalk-models/seq-fos-decon-$MODEL_SZ/seq-fos-$FOS-upsamp
