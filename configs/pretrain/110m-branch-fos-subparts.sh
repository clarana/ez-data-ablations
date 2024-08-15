export FOS=$1
export SUB_IND=$2
export NUM_SUBSETS=$3

# data (changes for other dataset classes)
export EXPT_BASE=subinds-fos
export SOURCE=s2orc-recombined
export EXPT=${EXPT_BASE}-part-$SOURCE-${FOS//[, ]/}-i${SUB_IND}-of${NUM_SUBSETS}

export PARTITION_INFO=/net/nfs.cirrascale/allennlp/claran/s2orc_partition_idx-9-27.json
export REMOTE=/net/nfs/allennlp/ananyaj/c4/
export LOCAL=/net/nfs.cirrascale/allennlp/claran/xppt-sources-branched/${SOURCE}/${FOS}/

export LOCAL_PARENT=/net/nfs.cirrascale/allennlp/claran/xppt-sources-branched/${SOURCE}/
export LOCAL_EVAL=/net/nfs.cirrascale/allennlp/claran/eval-perplexity/
export EVAL_GENERAL_PARTS=/net/nfs.cirrascale/allennlp/claran/xppt-sources-final/s2orc/

export CHECKPOINT_FOLDER=/data/claran/branch-xppt-${EXPT_BASE}/${SOURCE}/${FOS}-i${SUB_IND}
export SAVE_INTERVAL=20ba
export NUM_CHECKPOINTS_TO_KEEP=1

export GPUS=1

export BATCH_SIZE=1024
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
export MAX_DURATION=20500ba
#20500ba
# this is ~1B tokens
#20000ba
export WARMUP=0ba
#400ba
export DECAY_UNTIL=20000ba
#20000ba
export SEED=314159265

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

export HF_DATASETS_CACHE=/net/nfs.cirrascale/allennlp/claran/hf_cache/

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
    --partition_include "${FOS}," \
    --upsample \
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
