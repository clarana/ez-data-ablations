from data.streaming import StreamingDataset

from pathlib import Path
from typing import Any, Dict, List, Optional
import itertools
import re
import math
import numpy as np

from torch.utils.data import DataLoader, Subset, ConcatDataset
from composer.utils import dist

from .olmo_dependencies import OlmoConfigurationError, barrier, get_global_rank, PathOrStr
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset

import os

__all__ = ["MemMapDataset", "DataCollator", "IterableDataset", "build_eval_dataloader", "build_train_dataloader", "PathOrStr", "barrier", "get_global_rank"]

DATA_SOURCES={'s2orc-recombined':'/net/nfs.cirrascale/allennlp/claran/s2orc_partition_idx-9-27.json',
              'm2d2_wiki-toplevel':'/net/nfs.cirrascale/allennlp/claran/m2d2_wiki-top_partition_idx-3-19.json',
              'm2d2_wiki':'/net/nfs.cirrascale/allennlp/claran/m2d2_wiki_partition_idx-3-19.json'}

# max seq length
# paths or datasets (maybe just paths)
def build_memmap_dataset(args: Any, split: str = "train", data_dir: str = None, tgt_part=None, upsample=False, new_dataload=False,
                         parent_source=None, multiplier=None,#include_match=None, exclude_match=None, exact_match=None
                         ) -> MemMapDataset:
    paths: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None
    assert (parent_source is not None and multiplier is not None) or not new_dataload # need to specify explicitly with new dataloading mode
    split_d = {'train':'train', 'eval':'val', 'test':'test'}
    paths = []
    metadata = []
    train_multiple = split == "train" and hasattr(args, "partition_include") and (args.partition_include or args.partition_exclude or args.partition_name_list or upsample or (multiplier and multiplier != 1))
    print("data dir:", data_dir)
    if split == "train" and hasattr(args, "local_parts_parent"):
        print("args parent (from args):", os.path.join(args.local_parts_parent, parent_source))
    elif new_dataload and parent_source:
        print("args parent (parent source arg):", parent_source)
    if train_multiple:
        print("Training on multiple partitions of data")
    n_copies = 1 # unused unless training with upsampling
    
    if split not in ['train', 'eval']:
        raise OlmoConfigurationError("Building memmap dataset, split must be train or eval")
    if train_multiple:
        data_dir = os.path.join(args.local_parts_parent, parent_source) if new_dataload else args.local_parts_parent
        dir_name = data_dir.strip('/').split('/')[-1]
        if dir_name in DATA_SOURCES:
            source_names = [dir_name]
            all_candidate_partitions = [(dir_name, part) for part in os.listdir(data_dir)]
        else:
            source_names = args.source_names
            all_candidate_partitions = list(itertools.chain(
                *[[(source_name, part) for part in os.listdir(os.path.join(data_dir, source_name))] for source_name in
                  source_names]))

        print("Training data sources:", source_names)
        if upsample: # upsampling only makes sense when training over *multiple partitions* at once, also upsampling only occurs within a partition
            import pandas as pd
            n_copies_d = {}
            max_train_toks = 0
            for source in source_names:
                df = pd.read_json(DATA_SOURCES[source]).transpose()
                if df.train_toks.max() > max_train_toks:
                    max_train_toks = df.train_toks.max()

            for source in source_names:
                df = pd.read_json(DATA_SOURCES[source]).transpose()
                df['group_name'] = df['group_name'].str.replace(" ", "")
                df['n_copies'] = max_train_toks // df.train_toks
                n_copies_d[source] = df[['group_name', 'n_copies']].set_index('group_name').to_dict()['n_copies']
    else:
        dir_name = data_dir.strip('/').split('/')[-1]  # may or may not be one of the valid DATA_SOURCES
        print("dir_name:", dir_name, "| data_dir:", data_dir)
        all_candidate_partitions = [(dir_name, part) for part in os.listdir(data_dir)]

    for source, part in all_candidate_partitions:
        if source not in DATA_SOURCES:
            # print("source", source)
            pass
        if tgt_part and part != tgt_part:
            continue
        if split == "train" and hasattr(args, "partition_include") and not train_multiple and part != split:
            print(part, split)
            continue
        if train_multiple:
            if args.partition_include: # TODO: option for finer-grained (in|ex)clusion at the .npy level
                if not re.search(args.partition_include.lower(), part.lower()): # verified that ignoring case doesn't induce unexpected behavior with s2orc, but may have to edit later
                    continue
            if args.partition_exclude:
                if re.search(args.partition_exclude, part):
                    continue
            if args.partition_name_list:
                if part not in args.partition_name_list:
                    continue
            if (not new_dataload and args.upsample) or (new_dataload and upsample):
                n_copies = n_copies_d[source][part]
            if new_dataload and multiplier:
                n_copies *= multiplier
            print("ADDING", n_copies, "COPIES OF PARTITION", part, "TO TRAINING DATA")
        src_path = os.path.join(data_dir, part)
        print(src_path)
        if isinstance(source, str) and source not in src_path:
            src_path = os.path.join(data_dir, source, part)
        # TODO: MAKE SURE THIS WORKS WHEN VAL SPLIT MISSING
        print("Adding", split, "data from path", src_path)
        if any([spl in os.listdir(src_path) for spl in split_d.values()]): # os.listdir(src_path) -> .npy if single partition, else split dirs
            src_path = os.path.join(src_path, split_d[split])
        print("src_path:", src_path)
        if not os.path.exists(src_path):
            print("WARNING: src path", src_path, "does not exist")
            return # hacky, clean up later (hopefully breaks everything)
        metadata.extend([{'label':part}]*(len(os.listdir(src_path))*n_copies))
        for path in os.listdir(src_path):
            for i in range(n_copies):
                paths.append(os.path.join(src_path, path))
    if paths:
        print("Making MemMapDataset with paths", paths, end="\n\n")
        return MemMapDataset(*paths, chunk_size=args.seq_len, metadata=metadata)
    



# TODO: adapt to xp-pt codebase
def build_eval_dataloader(
    args: Any,
    collator: Any,
    batch_size: int,
    shuffle: bool = False,
    eval_dir: str = None,
    tgt_part: str = None,
    split: str = "eval",
) -> DataLoader:

    if eval_dir is None:
        eval_dir = args.local_eval

    dataset = build_memmap_dataset(args, split=split, data_dir=eval_dir, tgt_part=tgt_part)

    if dataset is None:
        return

    sampler = dist.get_sampler(dataset, shuffle=False, drop_last=False)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=bool(args.num_workers),#,True,
    )

MAX_DURATION_SEED_MODEL=20000

def build_train_dataloader(args: Any, collator: Any) -> DataLoader:
    if isinstance(args.max_duration, str) and "ba" in args.max_duration:
        total_dur_int = int(args.max_duration[:-2])
    else:
        total_dur_int = args.max_duration
    dur_int = total_dur_int if args.seed_load_weights_only else total_dur_int-MAX_DURATION_SEED_MODEL
    # dur_int = int(args.max_duration[:-2])
    total_seqs = dur_int * args.batch_size * args.n_gpu
    total_toks = total_seqs*args.seq_len # seq len 2048, total bs 1024
    print(f"Building train dataloader for optimization over {args.max_duration} ({total_seqs} sequences, or {total_toks} tokens)")
    if hasattr(args, "partition_rules_list") and args.partition_rules_list:
        print("multiple data partition rules!")
        datasets = []
        props_or_mults = []

        for rule in args.partition_rules_list:
            _,_,_,_,proportion_or_multiplier,_ = rule.split(':')
            proportion_or_multiplier = eval(proportion_or_multiplier)
            props_or_mults.append(proportion_or_multiplier)
            assert (isinstance(proportion_or_multiplier, int) or isinstance(proportion_or_multiplier,
                                                                            float)) and proportion_or_multiplier > 0
        if math.isclose(sum(props_or_mults), 1):
            print("Treating dataset weights as ratios or proportions")
            ds_weighting = "proportion"
        elif all([x >= 1 for x in props_or_mults]):  # TODO: figure out if they need to be ints as well
            print("Treating dataset weights as multipliers (ignores dataset sizes)")
            ds_weighting = "multiplier"
        else:
            raise OlmoConfigurationError(
                "Dataset weights must be proportions summing to 1 or multipliers for independently upweighting sources")  # TODO: figure out if int upweights necessary

        for rule in args.partition_rules_list:
            parent_source, include_match, exclude_match, exact_match, proportion_or_multiplier, upsamp_within = rule.split(':') # need to set up optional args or make a new version
            print(parent_source, include_match, exclude_match, exact_match, proportion_or_multiplier, upsamp_within)
            assert parent_source in DATA_SOURCES.keys()
            # match_rules = [include_match, exclude_match, exact_match]
            # for i in range(3):
            #     if match_rules[i] == "":
            #         match_rules[i] = None
            prop_or_mult = eval(proportion_or_multiplier)
            upsamp_within = upsamp_within.lower() == "true"
            args.partition_include = include_match
            args.partition_exclude = exclude_match
            ds = build_memmap_dataset(args, split="train", data_dir=os.path.join(args.local_parts_parent, parent_source, exact_match) if exact_match else "",
                                      new_dataload=args.new_dataload,
                                      parent_source=parent_source,
                                      multiplier=prop_or_mult if ds_weighting == "multiplier" else 1,
                                      upsample=upsamp_within)
            # ds = build_memmap_dataset(args, split="train", data_dir=args.local, new_dataload=args.new_dataload,
            #                           data_dir=parent_source,
            #                           include_match=include_match if include_match else None,
            #                           exclude_match=exclude_match if exclude_match else None,
            #                           exact_match=exact_match if exact_match else None,
            #                           multiplier=prop_or_mult if ds_weighting == "multiplier" else 1,
            #                           upsample=upsamp_within)

            datasets.append({'rule': rule, 'memmap_dataset':ds, 'prop_or_mult': prop_or_mult, 'ds_len': len(ds)})

        print(datasets)
        total_ds_size = sum([d['ds_len'] for d in datasets])
        print("total # sequences:", total_seqs)
        dataset_subsets = []
        for dset in datasets:
            if ds_weighting == "proportion":
                ds_mix_seqs = round(dset['prop_or_mult'] * total_seqs)  # estimated number of sequences in data mixture that should belong to each dataset based on specified proportions
                subset_inds = np.random.choice(range(dset['ds_len']), ds_mix_seqs)
                subset = Subset(dset['memmap_dataset'], subset_inds)
                print("resampling", dset['rule'], "dataset to", ds_mix_seqs, "sequences")
                dataset_subsets.append(subset)
            else:
                dataset_subsets.append(dset['memmap_dataset'])
        dataset = ConcatDataset(dataset_subsets)


        # for dataset in datasets:
        #     ds_data_prop = dataset['ds_len'] / total_ds_size # proportion of data pool belonging to each dataset/source
        #     ds_mix_prop = dataset['prop_or_mult'] * dur_int # number

    else:
        dataset = build_memmap_dataset(args, split="train", data_dir=args.local, upsample=args.upsample if hasattr(args, "upsample") else False, new_dataload=args.new_dataload if hasattr(args, "new_dataload") else False)

    print(len(dataset))

    work_dir = Path(args.checkpoint_folder) / "train_data"
    if get_global_rank() == 0:
#         if work_dir.is_dir() and not args.save_overwrite:
#             raise OlmoConfigurationError(
#                 "train data working directory already exists, use --save_overwrite to overwrite"
#             )
#         else:
        work_dir.mkdir(exist_ok=True, parents=True)
    barrier()
    if isinstance(args.max_duration, str) and "ba" in args.max_duration:
        dur_int = int(args.max_duration[:-2])
    else:
        dur_int = args.max_duration
    
    if args.subset_ind != -1:
        print(f"subsampling dataset w/ 1/{args.num_subsets}:", args.subset_ind)
        dataset = Subset(dataset, range(args.subset_ind, len(dataset), args.num_subsets))
        
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            seed=args.seed,
            shuffle=True,
            drop_last=True,
            max_examples=args.batch_size * args.n_gpu * dur_int, # NOTE: commit using MAX_DURATION_SEED_MODEL in calculation of dur_int fixes a previous possible issue
            work_dir=work_dir,
        ),
        batch_size=args.batch_size,
        drop_last=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=bool(args.num_workers),
    )
