export FOS1=$1
export ALL_L1="Culture_and_the_arts|General_referece|Health_and_fitness|History_and_events|Human_activites|Mathematics_and_logic|Natural_and_physical_sciences|Philosophy_and_thinking|Religion_and_belief_systems|Society_and_social_sciences|Technology_and_applied_sciences"
export L1=$2
export IND=$3
export GPUS=$4
export NUM_BIL_TOKS=$5
export MODEL_SZ=$6

wandb login --relogin $WANDB_KEY

export EXPT_BASE=data_mix_fos_L1_upsamp
function join_by { local IFS="$1"; shift; echo "$*"; }
export SOURCES=("s2orc-recombined" "m2d2_wiki-toplevel")

export EXPT=${EXPT_BASE}-$(join_by "-" "${SOURCES[@]}")-i${IND}-${FOS1//[, ]/}-${L1//[, ]/}

export PARTITION_INFO=/net/nfs.cirrascale/allennlp/claran/s2orc_partition_idx-9-27.json
export REMOTE=/net/nfs/allennlp/ananyaj/c4/

export LOCAL_PARENT=/net/nfs.cirrascale/allennlp/claran/xppt-sources-branched/${SOURCE}/
export LOCAL_EVAL=/net/nfs.cirrascale/allennlp/claran/eval-perplexity/
export EVAL_GENERAL_PARTS=/net/nfs.cirrascale/allennlp/claran/xppt-sources-final/s2orc/


export CHECKPOINT_FOLDER=/results/branch-xppt-${EXPT_BASE}/${SOURCE}/$EXPT
export SAVE_INTERVAL=50ba
export NUM_CHECKPOINTS_TO_KEEP=1


# system
# IMP: batch_size needs to change if num GPUs change
# micro_batch_size is per GPU, that remains constant
# batch_size * gpus = 1024 (1024 sequence each with len 2048 = ~2M tokens/batch)
# micro_batch_size is number of sequences that fit per GPU at once (rest is grad accumulation)
export BATCH_SIZE=$((1024/${GPUS}))
#128 # for 8 gpus, but single gpu needs to be 1024
[[ $MODEL_SZ = 110m ]] && export MICRO_BATCH_SIZE=16 || export MICRO_BATCH_SIZE=8
export NUM_WORKERS=2

export PREFETCH_FACTOR=16

# optim
# LR remains same for AdamW, WD changes to 0.1
export OPTIM=adamw
#lionw
export LR=6e-5
export WEIGHT_DECAY=0.1

export MAX_DURATION=$((20000 + (500*${NUM_BIL_TOKS})))ba
export WARMUP=0ba
export DECAY_UNTIL=20000ba
export SEED=314159265

export EVAL_INTERVAL=500ba

export EVAL_SUBSET_NUM_BATCHES=100
export EVAL_BATCH_SIZE=8

export MODEL_PATH=/net/nfs.cirrascale/allennlp/claran/seed-xppt-decon-${MODEL_SZ}/customgpt-${MODEL_SZ//_/.}-tokens-41943040000-rank-0
export MODEL_NAME=customgpt-${MODEL_SZ//_/.}
export TOKENIZER=allenai/gpt-neox-olmo-dolma-v1_5
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/net/nfs.cirrascale/allennlp/claran/hf_cache

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
    --partition_include "(${FOS1},|${L1}$)"\
    --new_dataload \
    --upsample \
    --source_names "${SOURCES[@]}" \
    --prefetch_factor $PREFETCH_FACTOR \
    --checkpoint_folder $CHECKPOINT_FOLDER \
    --seed $SEED \
    --n_gpu $GPUS \
    --eval_ood_partitions \
    --eval_subset_num_batches $EVAL_SUBSET_NUM_BATCHES \
    --resume_training \
    --ood_eval_only \
    --skip_downstream \
    --eval_general_partitions $EVAL_GENERAL_PARTS \
    --eval_perplexity_subsets > >(tee -a "/results/${EXPT}-stdout.log") 2> >(tee -a "/results/${EXPT}-stderr.log" >&2)

cp "$CHECKPOINT_FOLDER/$EXPT-metrics.json" /results/metrics.json

export checkpt=$(find $CHECKPOINT_FOLDER | grep customgpt-${MODEL_SZ//_/.}-tokens)
echo "$checkpt"

if [ $(echo "$checkpt" | wc -l) -gt 1 ]; then
    last_checkpt=$(echo "$checkpt" | awk -F- '{split($0, b); print b[length(b)-2]}' | sort -n | tail -n 1);
    checkpt=$(echo "$checkpt" | grep $last_checkpt);
fi

python xp-pt-as-olmo/convert_checkpoint.py --checkpoint $checkpt --config xp-pt-as-olmo/configs/customgpt-${MODEL_SZ}.yaml --output_dir /net/nfs.cirrascale/allennlp/claran/catwalk-models/data-mix-upsamp-s2orc-wiki-$MODEL_SZ/seq-fos-L1-i${IND}-${FOS1//[, ]/}-${L1//[, ]/}
cp /net/nfs.cirrascale/allennlp/claran/catwalk-models/seed-decon-5epoch-${MODEL_SZ}/*.json /net/nfs.cirrascale/allennlp/claran/catwalk-models/data-mix-upsamp-s2orc-wiki-$MODEL_SZ/seq-fos-L1-i${IND}-${FOS1//[, ]/}-${L1//[, ]/}
