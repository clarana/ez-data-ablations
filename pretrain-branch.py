import argparse
import transformers
import torch
import time
import torch.nn as nn
import torch.distributed as dist

from copy import deepcopy
from composer import Trainer
from composer import Callback, State, Logger
from composer.algorithms import GradientClipping

from composer.callbacks import (
    CheckpointSaver,
    SpeedMonitor,
    LRMonitor,
    OptimizerMonitor,
    MemoryMonitor,
)
from composer.core import Evaluator, DataSpec
from composer.loggers import WandBLogger, InMemoryLogger
from composer.optim import ConstantScheduler

from composer import Callback, State, Logger

from itertools import chain
from torch.utils.data import DataLoader

from eval_suite import create_eval_suite
from callbacks import WandBMetrics

from model import create_customgpt
from model.customgpt import RMSLayerNorm, LayerNorm
from optim import DecoupledLionW
from data import build_train_dataloader, build_eval_dataloader, DataCollator, DATA_SOURCES

from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from enum import Enum
import pdb
import os
import json

class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"

class PaddingDirection(StrEnum):
    right = "right"
    left = "left"


# create a keyvalue class
class keyvalue(argparse.Action):
    # Constructor calling
    def __call__( self , parser, namespace,
                 values, option_string = None):
        setattr(namespace, self.dest, dict())

        # works for simple int, float, bool
        # does not convert 0 and 1 to bool
        # if the downstream param expectation is that of a bool
        # also cannot handle 1e4 notation as of now
        def convert_val(value):
            if value[0].isdigit():
                # either float or int
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            elif value == 'True' or value == 'False':
                # convert to bool
                if value == 'True':
                    value = True
                else:
                    value = False

            return value

        for value in values:
            # split it into key and value
            key, value = value.split('=')
            # assign into dictionary
            getattr(namespace, self.dest)[key] = convert_val(value)
            
            
class MetricsSaver(Callback):
    
    def __init__(self, args, in_mem_logger):
        self.logger = in_mem_logger
    
    def eval_end(self, state: State, logger: Logger) -> None:
        d = {}
        for k, v in self.logger.most_recent_values.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.item()
            else:
                d[k] = v
        with open(os.path.join(args.checkpoint_folder, f"{args.exp_name}-metrics-{state.timestamp.batch}.json"), "w") as f:
            json.dump(d, f, indent=6)

def get_latest_checkpoint(args):
    if args.resume_training:
        print("Getting latest checkpoint")
    actual_checkpoint_path = os.path.join(args.checkpoint_folder, args.exp_name)
    
    if not args.resume_training or not os.path.exists(actual_checkpoint_path):
        print("Starting from seed model, or checkpoint path", actual_checkpoint_path, "not found")
        return args.model_path
    
    if os.path.exists(os.path.join(actual_checkpoint_path, "latest-rank0.pt")):
        print("Loading model from", os.path.join(actual_checkpoint_path, "latest-rank0.pt"))
        return os.path.join(actual_checkpoint_path, "latest-rank0.pt")
    
    
    checkpts = [file for file in os.listdir(actual_checkpoint_path) if 'rank-0' in file]
    
    if len(checkpts) == 0:
        return args.model_path
    
    highest_rank = max([int(pt.split('-')[-3]) for pt in checkpts])
    latest_pt = [pt for pt in checkpts if str(highest_rank) in pt][0]
    return os.path.join(actual_checkpoint_path, latest_pt) 


def main(args):
    # TODO: make sure this is multi-GPU training aware, ie takes care of splitting samples across GPUs and sampling
    # Initialize process group and set device.
    if args.debug:
        args.num_workers=0
        args.prefetch_factor=None
        # also be sure to run export GPUS=1 before running the run command
    
    
    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True) # Q: use fast?
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    # collate function
    collate_fn = DataCollator(pad_direction=PaddingDirection.right, pad_token_id=tokenizer.eos_token_id)

    # dataloader
    train_dataloader = build_train_dataloader(
        args,
        collator=collate_fn,
        
    )

    eval_dataloader = build_eval_dataloader(
        args,
        collator=collate_fn,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    
    # dataspec
    def tokens_per_batch(batch: Dict) -> int:
        # batch size * seq_len
        return batch['input_ids'].shape[0] * batch['input_ids'].shape[1]
    
    def samples_per_batch(batch: Dict) -> int:
        return batch['input_ids'].shape[0]

    train_dspec = DataSpec(
        dataloader=train_dataloader,
        get_num_tokens_in_batch=tokens_per_batch,
        get_num_samples_in_batch=samples_per_batch,
    )
    
    eval_dspec = DataSpec(
        dataloader=eval_dataloader,
        get_num_tokens_in_batch=tokens_per_batch,
        get_num_samples_in_batch=samples_per_batch,
    )

    c4_evaluator = Evaluator(
        label='eval',
        dataloader=eval_dspec,#eval_dataloader,
        metric_names=["Perplexity", "CrossEntropy"],
        subset_num_batches=args.eval_subset_num_batches,    # only set this for C4 LM eval
    ) # different eval_dspec for eval?

    # per-domain perplexity evals
    domain_evaluators = []
    if args.eval_perplexity_subsets:
        for part in os.listdir(args.local_eval):
            domain_eval_dataloader = build_eval_dataloader(
                args,
                collator=collate_fn,
                batch_size=args.eval_batch_size,
                shuffle=False,
                tgt_part=part
            )
            domain_eval_dspec = DataSpec(
                dataloader=domain_eval_dataloader,
                get_num_tokens_in_batch=tokens_per_batch,
                get_num_samples_in_batch=samples_per_batch,
            )

            domain_evaluator = Evaluator(
                label=f'eval-{part}',
                dataloader=domain_eval_dspec,  # eval_dataloader,
                metric_names=["Perplexity", "CrossEntropy"],
                subset_num_batches=None,#args.eval_subset_num_batches,  # only set this for C4 LM eval
            )  # different eval_dspec for eval?
            domain_evaluators.append(domain_evaluator)

    if args.eval_ood_partitions:
        if args.new_dataload and args.partition_rules_list:
            eval_dirs = [os.path.join(args.local_parts_parent, x.split(':')[0]) for x in args.partition_rules_list]
        elif args.local_parts_parent.strip('/').split('/')[-1] not in DATA_SOURCES:
            eval_dirs = [os.path.join(args.local_parts_parent, source) for source in args.source_names]
        else:
            eval_dirs = [args.local_parts_parent]
        print(eval_dirs)
        for eval_dir in eval_dirs:
            for part in os.listdir(eval_dir):
                for spl in ['train', 'eval']:
                    if spl == "train" and args.ood_eval_only:
                        continue

                    domain_eval_dataloader = build_eval_dataloader(
                        args,
                        collator=collate_fn,
                        batch_size=args.eval_batch_size,
                        shuffle=False,
                        tgt_part=part,
                        split=spl,
                        eval_dir=eval_dir,
                    )
                    if domain_eval_dataloader is None:
                        print("skipping ood eval set", part, "due to empty", spl, " split")
                        continue
                    domain_eval_dspec = DataSpec(
                        dataloader=domain_eval_dataloader,
                        get_num_tokens_in_batch=tokens_per_batch,
                        get_num_samples_in_batch=samples_per_batch,
                    )

                    domain_evaluator = Evaluator(
                        label=f'eval-ood-{part}-{spl}',
                        dataloader=domain_eval_dspec,  # eval_dataloader,
                        metric_names=["Perplexity", "CrossEntropy"],
                        subset_num_batches=args.eval_subset_num_batches,  # only set this for C4 LM eval
                    )  # different eval_dspec for eval?
                    domain_evaluators.append(domain_evaluator)
        
    if args.eval_general_partitions:
        general_domain_eval_dataloader = build_eval_dataloader(
            args,
            collator=collate_fn,
            batch_size=args.eval_batch_size,
            shuffle=False,
            split="eval",
            eval_dir=args.eval_general_partitions,
        )
        general_domain_eval_dspec = DataSpec(
            dataloader=general_domain_eval_dataloader,
            get_num_tokens_in_batch=tokens_per_batch,
            get_num_samples_in_batch=samples_per_batch,
        )
        general_domain_evaluator = Evaluator(
            label=f'eval-general-domain',
            dataloader=general_domain_eval_dspec,  # eval_dataloader,
            metric_names=["Perplexity", "CrossEntropy"],
            subset_num_batches=args.eval_subset_num_batches,
        )
        domain_evaluators.append(general_domain_evaluator)


    model, prev_params_for_flops = create_customgpt(
        model_name=args.model_name,
        tokenizer=tokenizer,
        model_path=None if args.model_path == '' else args.model_path,
        model_config=args.model_config,
    )

    # optimizer and scheduler
    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    elif args.optim == 'lionw':
        decay = set()
        no_decay = set()
        all_params = {}

        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                # NOTE: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times, but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                all_params[fpn] = p

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, nn.Linear):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, (RMSLayerNorm, LayerNorm, nn.LayerNorm, nn.Embedding)):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # Validate that we've considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert (
            len(all_params.keys() - union_params) == 0
        ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

        # Create the pytorch optimizer groups.
        optim_groups = [
            {"params": [all_params[pn] for pn in sorted(list(decay))], "weight_decay": args.weight_decay},
            {"params": [all_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = DecoupledLionW(optim_groups, lr=args.lr, betas=(args.beta1, args.beta2))

    scheduler = ConstantScheduler(alpha=0.1 if args.load_optim_state else 1.0)

    fsdp_config = None

    if args.fsdp:
        fsdp_config = {
            'sharding_strategy': 'FULL_SHARD',
            'cpu_offload': False,
            'mixed_precision': 'PURE',
            'limit_all_gathers': True,
            'activation_checkpointing': False,
            'activation_cpu_offload': False,
            'activation_checkpointing_reentrant': False,
            'use_orig_params': True,  # needed to work with compile
            'verbose': True,
        }

    # eval suite
    eval_suite = create_eval_suite(tokenizer, args.eval_batch_size)

    # loggers
    wandb_logger = WandBLogger(
        project='xp-pt',	
        entity='allennlp',	
        name=args.exp_name,
    )
    
    in_mem_logger = InMemoryLogger()
            
    

    # add algorithms
    algorithms = []

    algorithms.append(GradientClipping(clipping_type='norm', clipping_threshold=args.grad_clipping_threshold))
    wandb_plotting_callback = WandBMetrics(num_params_for_flops=model.num_params_for_flops)

    # with deepspeed, checkpoints are saved for every rank
    # including rank var in necessary to prevent crashing
    checkpoint_saver = CheckpointSaver(
        folder=args.checkpoint_folder + '/{run_name}',
        filename=args.model_name.replace('/', '-') + '-tokens-{token}-rank-{rank}',
        save_interval=args.save_interval,
        num_checkpoints_to_keep=args.num_checkpoints_to_keep,
    )

    # TODO: verify optimizer states saved properly with/without fsdp
    callbacks = [
        checkpoint_saver,
        SpeedMonitor(gpu_flops_available=312e12),   # A100 doc from mosaicml
        LRMonitor(),
        OptimizerMonitor(),
        MemoryMonitor(),
        wandb_plotting_callback,
        MetricsSaver(args, in_mem_logger),
    ]

    if args.load_optim_state:
        optimizer.load_state_dict(state['optimizers'][optimizer.__class__.__name__])

    actual_checkpoint_path = os.path.join(args.checkpoint_folder, args.exp_name)
    
    # os.path.join(actual_checkpoint_path, "latest-rank0.pt") if (args.resume_training and os.path.exists(os.path.join(actual_checkpoint_path, "latest-rank0.pt"))) else args.model_path,
    # args.seed_load_weights_only and not os.path.exists(os.path.join(actual_checkpoint_path, "latest-rank0.pt")), # load only seed weights (no optimizer state) if 
    latest_checkpoint = get_latest_checkpoint(args)
    
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        schedulers=scheduler,
        algorithms=algorithms,
        train_dataloader=train_dspec,
        eval_dataloader=[c4_evaluator] + domain_evaluators + (eval_suite if not args.skip_downstream else []),
        eval_interval=0 if args.eval_interval == '' else args.eval_interval,
        device='gpu',
        max_duration=args.max_duration,
        precision=args.precision,
        device_train_microbatch_size=args.device_train_microbatch_size,
        seed=int(time.time()) if args.seed is None else args.seed,
        run_name=args.exp_name,
        callbacks=callbacks,
        loggers=[wandb_logger, in_mem_logger],
        progress_bar=True,
        fsdp_config=fsdp_config,
        save_overwrite=args.save_overwrite,
        load_path=latest_checkpoint,
        load_weights_only=args.seed_load_weights_only and (latest_checkpoint == args.model_path),
    ) # TODO: change to latest .pt??
    
#     breakpoint()
    trainer.fit()
    
    d = {}
    for k, v in in_mem_logger.most_recent_values.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.item()
            # cpu().numpy()
        else:
            d[k] = v
    print(d)
    with open(os.path.join(args.checkpoint_folder, f"{args.exp_name}-metrics.json"), "w") as f:
        json.dump(d, f, indent=6)
        
#     print(in_mem_logger.most_recent_values)
#     with open(os.path.join(args.checkpoint_folder, f"{args.exp_name}-metrics.json"), "w") as f:
#         json.dump(in_mem_logger.most_recent_values, f, indent=6) #todo: append seed model log?

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='', help='experiment name')
    parser.add_argument('--save_overwrite', action='store_true')

    # data params
    parser.add_argument('--remote', type=str, default='', help='remote location of C4 dataset')
    parser.add_argument('--local', type=str, default='', help='local cache directory (train data, .npy memmap data)')
    parser.add_argument('--source_names', nargs="*", help="add all sources to be trained on, contained within local_parts_parent directory. If used, include/exclude rules are applied after combining lists of partitions from all sources")
    parser.add_argument('--partition_idx', type=int, default=128, help='partition index')
    parser.add_argument('--partition_name', type=str, default="", help='partition name')
    parser.add_argument('--partition_name_list', nargs="*", help="add all partitions listed as arguments. If used, applied after partition_include (and partition_exclude) rules")
    parser.add_argument('--partition_rules_list', nargs="*", help="format for each: 'parent_source:include_match:exclude_match:exact_match:proportion_or_multiplier:upsamp_within'. \n\te.g. `s2orc-recombined:Biology,:2022::0.5:False s2orc-recombined:::Biology,2022-2022:0.5:False` Ignores empty match rules. (empty `include` == `*`). 0.0 <= prop <= 1.0 with sum = 1, or 1 <= multiplier")
    parser.add_argument('--partition_include', type=str, default="", help='add partition to training data if it includes this substring. Custom data mix proportions unsupported')
    parser.add_argument('--partition_exclude', type=str, default="", help='do not add partition to training data if it includes this substring')
    parser.add_argument('--upsample', action='store_true', help='use additional copies of underrepresented data automatically using partition info jsons') # needs partition_info to work
    parser.add_argument('--subset_ind', type=int, default=-1, help='if provided, index of training data subset. If num_partitions == 8, then pick from 0 to 7 inclusive')
    parser.add_argument('--num_subsets', type=int, default=8, help='if provided along with subset_ind, number of training data subsets total. Eighths by default.')
    parser.add_argument('--new_dataload', action='store_true', help='new (training) dataloading paradigm.')

    parser.add_argument('--partition_info', type=str, default="/net/nfs.cirrascale/allennlp/claran/s2orc_partition_idx-9-27.json", help="deprecating, unused. Hardcoded paths per source in data/__init__.py")
    
    parser.add_argument('--local_parts_parent', type=str, default='', help='if applicable, directory for all source data to evaluate on')
    parser.add_argument('--local_eval', type=str, default='/data/claran/fake-eval', help='local eval data directory')

    parser.add_argument('--model_name', type=str, default='customgpt-110m')
    parser.add_argument('--model_path', type=str, default='', help='specify model path to load model from check points instead of pre-trained')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--load_optim_state', action='store_true')

    parser.add_argument('--model_config', nargs='*', action=keyvalue)
    parser.add_argument('--tokenizer_name', type=str, default='gpt2')

    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--fsdp', action='store_true')

    parser.add_argument('--precision', type=str, default='amp_bf16')
    parser.add_argument('--lr', type=float, default=3e-4)   # max lr after warmup
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)

    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--prefetch_factor', type=int, default=8)

    parser.add_argument('--grad_clipping_threshold', type=float, default=1.0, help='clipping threshold for grad norm')
    parser.add_argument('--max_duration', type=str, default='20000ba')
    parser.add_argument('--t_warmup', type=str, default='400ba')
    parser.add_argument('--t_max', type=str, default='160000ba')
    parser.add_argument('--alpha_f', type=float, default=0.1, help="lr multiplier to decay to")

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--device_train_microbatch_size', type=int, default=1, help='microbatch size')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--n_gpu', type=int, default=0, help='number of gpus')
    parser.add_argument('--seed_load_weights_only', action='store_true', help="load only weights from seed model, no state")

    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.set_defaults(gradient_checkpointing=False)
    

    # eval
    parser.add_argument('--eval_interval', type=str, default='', help="after how much time we run an eval loop, blank string to disable running eval")
    parser.add_argument('--eval_subset_num_batches', type=int, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=4, help='separate batch size param, eval does not do grad accum')
    parser.add_argument('--save_interval', type=str, default="500ba", help='create checkpoints after')
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoints', help='checkpoint folder')
    parser.add_argument('--num_checkpoints_to_keep', type=int, default=3, help='number of checkpoints to keep')
    parser.add_argument('--eval_perplexity_subsets', action='store_true', help='eval perplexity suite is from multiple sources')
    parser.add_argument('--eval_ood_partitions', action='store_true', help='eval perplexity on other partitions from data source')
    parser.add_argument('--skip_downstream', action='store_true', help="run only perplexity evals; skip downstream tasks")
    parser.add_argument('--eval_general_partitions', type=str, default="", help='eval (sample of) all eval data')
    
    parser.add_argument('--debug', action='store_true', help='disable distributed training to enable pdb, override set parameters')
    parser.add_argument('--ood_eval_only', action='store_true', help='skip train partitions, only evaluate on eval splits for ood partition eval')
    args = parser.parse_args()
    main(args)
