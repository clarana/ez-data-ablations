# data (changes for other dataset classes)
export EXPT=evalonly-5epoch-update
#${1}
echo $EXPT
export REMOTE=/net/nfs/allennlp/ananyaj/c4/
export LOCAL=/data/seed-pretrain/
#export LOCAL_EVAL=/data/claran/fake-eval/
export SOURCE=s2orc-recombined
export LOCAL_PARENT=/net/nfs.cirrascale/allennlp/claran/xppt-sources-branched/${SOURCE}/
export LOCAL_EVAL=/net/nfs.cirrascale/allennlp/claran/eval-perplexity/
export EVAL_GENERAL_PARTS=/net/nfs.cirrascale/allennlp/claran/xppt-sources-final/s2orc/
# oops not really local but...


# system
# IMP: batch_size needs to change if num GPUs change
# micro_batch_size is per GPU, that remains constant
# batch_size * gpus = 1024 (1024 sequence each with len 2048 = ~2M tokens/batch)
# micro_batch_size is number of sequences that fit per GPU at once (rest is grad accumulation)
export GPUS=1

export NUM_WORKERS=2
#4

export PREFETCH_FACTOR=16

export SEED=15213

export EVAL_SUBSET_NUM_BATCHES=100
export EVAL_BATCH_SIZE=8

export MODEL_NAME=customgpt-110m
export MODEL_PATH=/net/nfs.cirrascale/allennlp/claran/seed-xppt/customgpt-110m-tokens-41943040000-rank-0
#/data/claran/seed-xppt/customgpt-110m-seed-pretrain/$1
#export MODEL_PATH=/data/claran/seed-xppt/$1
export TOKENIZER=allenai/eleuther-ai-gpt-neox-20b-pii-special
export TOKENIZERS_PARALLELISM=false

composer -n $GPUS --stdout stdout_{rank}.log --stderr stderr_{rank}.log --master_port 1234 eval-model.py \
    --exp_name $EXPT-$MODEL_NAME-seed-pretrain \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --tokenizer_name $TOKENIZER \
    --remote $REMOTE \
    --num_workers $NUM_WORKERS \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --local $LOCAL \
    --local_eval $LOCAL_EVAL \
    --local_parts_parent $LOCAL_PARENT \
    --prefetch_factor $PREFETCH_FACTOR \
    --seed $SEED \
    --n_gpu $GPUS \
    --eval_ood_partitions \
    --eval_subset_num_batches $EVAL_SUBSET_NUM_BATCHES \
    --ood_eval_only \
    --eval_general_partitions $EVAL_GENERAL_PARTS \
    --eval_perplexity_subsets > >(tee -a $EXPT-stdout.log) 2> >(tee -a $EXPT-stderr.log >&2)
