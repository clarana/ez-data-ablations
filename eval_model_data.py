import argparse
import transformers
import torch
import itertools
from copy import deepcopy

from evaluator import get_eval_data, get_model, get_tokenizer, run_log_likelihood, get_ppl_scores, get_full_model_path, reset_ppl_metrics

PROBLEM_PARTITIONS=["Culture_and_the_arts__Visual_arts", "Mathematics_and_logic__Mathematics"]

def main(args):
    if not args.full_model_paths:    
        full_model_paths = [get_full_model_path(model_name, args.catwalk_model_subdirs) for model_name in args.model_names]
    else:
        full_model_paths = args.full_model_paths
    
    tokenizer = get_tokenizer(full_model_paths[0])

    partition_names = [name for name in args.partition_names if not(any([prob in name for prob in PROBLEM_PARTITIONS]))] # filter out problem eval partitions
    
    eval_datasets = [get_eval_data(tokenizer, partition_name, data_subdir=args.data_subdir, split=args.split) for partition_name in partition_names]

        
    if args.model_eval_mode == "eval_only":
        for full_model_path in full_model_paths:
            model = get_model(full_model_path).to(f'cuda:{args.gpu_ind}')
            for partition_name, eval_data in zip(partition_names, eval_datasets):
                ppl_metric, doc_ppl_scores = get_ppl_scores(model, tokenizer, **eval_data)
                
                if args.data_eval_mode == "each":
                    print(full_model_path, partition_name, end=":\n")
                    print(ppl_metric)
                    reset_ppl_metrics()
                # todo, save somewhere
            if args.data_eval_mode == "combine":
                print(full_model_path, partition_name, end=":\n")
                print(ppl_metric)
                reset_ppl_metrics()
    elif args.model_eval_mode == "uniform_avg_eval":
        assert len(full_model_paths) > 0, "Must provide more than one model if parameter averaging"
        
        # create uniform parameter average of provided models
        averageable_state_dicts = []
        for i, full_model_path in enumerate(full_model_paths):
            part_model = transformers.models.auto.AutoModelForCausalLM.from_pretrained(full_model_path).to(f'cuda:{args.gpu_ind}')
            if i == 0:
                interpol_dict = deepcopy(part_model.state_dict())
                divisor = torch.div(1, len(full_model_paths))
            else:
                averageable_state_dicts.append(part_model.state_dict())
        ref_keys = list(interpol_dict.keys())
        for tgt_sd in averageable_state_dicts:
            tgt_keys = list(tgt_sd.keys())
            assert ref_keys == tgt_keys

        for param_name in ref_keys:
            param_data = interpol_dict[param_name].data.clone()

            # Skip the int64 components which don't need averaging
            if not param_data.is_floating_point():
                continue

            for a_sd in averageable_state_dicts:
                param_data.add_(a_sd[param_name].data)
            param_data.mul_(divisor)
            interpol_dict[param_name] = param_data

        new_sd = {}
        for key in interpol_dict:
            new_key = key#[6:] # drop the "model." part of the param name
            new_sd[new_key] = interpol_dict[key]
            
        # evaluate parameter averaged model
        part_model.load_state_dict(new_sd) # just reuse bc it exists
        for partition_name, eval_data in zip(partition_names, eval_datasets):
            ppl_metric, doc_ppl_scores = get_ppl_scores(part_model, tokenizer, **eval_data)
            if args.data_eval_mode == "each":
                print(full_model_paths, partition_name, end=":\n")
                print(ppl_metric)
                reset_ppl_metrics()
        if args.data_eval_mode == "combine":
            print(full_model_paths, partition_name, end=":\n")
            print(ppl_metric)
            reset_ppl_metrics()

    print("----------")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu_ind', default=0, help="if multiple gpus available, pick one to run evals on")
    # mode arguments
    parser.add_argument('--model_eval_mode', default="eval_only", choices=["eval_only", "uniform_avg_eval"], help="If multiple models provided, whether or not to form uniform parameter average for evaluation") # TODO, non-uniform weighted averaging, read in data similarity matrix
    parser.add_argument('--data_eval_mode', default="each", choices=["each", "combine"], help="If multiple data subdirs, whether to evaluate each separately or in aggregate")
    
    # model arguments
    parser.add_argument('--model_names', nargs="*", help='model name (assume stored in catwalk model home)')
    parser.add_argument('--catwalk_model_subdirs', nargs="*", default=[], help="subdirectories (in order). Use only if all models are contained in the same subdir")
    
    parser.add_argument('--full_model_paths', nargs="*", default=[], help="multiple full model paths (assume stored in catwalk model home). Use when models not in same subdir")
    
    # data arguments
    parser.add_argument('--partition_names', nargs="+", help="eval data partition name (assume stored in catwalk data home")
#     parser.add_argument('--partition_name_list', nargs="*", help="TODO: partition names (assume stored in catwalk data home). Adds all partitions listed as arguments. If used, applied after partition_include (and partition_exclude) rules")
    parser.add_argument('--data_subdir', type=str, default="xppt-s2orc-ppl", help='partition data subdirectory (assumed lives in catwalk data home')
    parser.add_argument('--split', type=str, default="val", choices=['val', 'test'], help="split for eval")
    
    # result arguments
    # TODO, where/how to save
    
    # soup arguments
    
    
    args = parser.parse_args()
    main(args)
