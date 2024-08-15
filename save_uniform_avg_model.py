import argparse
import transformers
import torch
import itertools
from copy import deepcopy

from evaluator import get_eval_data, get_model, get_tokenizer, run_log_likelihood, get_ppl_scores, get_full_model_path, reset_ppl_metrics
import os
import shutil

def main(args):
    if not args.full_model_paths:    
        full_model_paths = [get_full_model_path(model_name, args.catwalk_model_subdirs) for model_name in args.model_names]
    else:
        full_model_paths = args.full_model_paths

    print("Loading", len(full_model_paths), "component models:", full_model_paths)

    tokenizer = get_tokenizer(full_model_paths[0])

    assert len(full_model_paths) > 0, "Must provide more than one model if parameter averaging"

    # create uniform parameter average of provided models
    averageable_state_dicts = []
    for i, full_model_path in enumerate(full_model_paths):
        part_model = transformers.models.auto.AutoModelForCausalLM.from_pretrained(full_model_path).to('cpu')
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
        new_key = key[6:] # drop the "model." part of the param name
        new_sd[new_key] = interpol_dict[key]

    if not args.full_output_path:
        save_path = get_full_model_path(args.output_model_name, args.output_subdirs)
    else:
        save_path = args.full_output_path

    print("\tSaving uniform averaged model to:", save_path)
    os.makedirs(save_path, exist_ok=True)
    torch.save(new_sd, os.path.join(save_path, "model.pt"))

    seed_model_path = get_full_model_path(f"seed-decon-5epoch-{args.model_size}")
    model_json_cfg_files = [file for file in os.listdir(seed_model_path) if 'json' in file or 'config' in file]

    for file in model_json_cfg_files:
        shutil.copy(os.path.join(seed_model_path, file), save_path)


    print("----------")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # model arguments
    parser.add_argument('--model_names', nargs="*", help='model name (assume stored in catwalk model home)')
    parser.add_argument('--catwalk_model_subdirs', nargs="*", default=[], help="subdirectories of component models (in order). Use only if all models are contained in the same subdir")
    
    parser.add_argument('--full_model_paths', nargs="*", default=[], help="multiple full model paths (assume stored in catwalk model home). Use when models not in same subdir")
    
    # result arguments
    parser.add_argument('--output_model_name', type=str, help="eval data partition name (assume store in catwalk model home")
#     parser.add_argument('--partition_name_list', nargs="*", help="TODO: partition names (assume stored in catwalk data home). Adds all partitions listed as arguments. If used, applied after partition_include (and partition_exclude) rules")
    parser.add_argument('--output_subdirs', nargs="*", help='subdirectory of output model (assume write to catwalk model home')
    parser.add_argument('--full_output_path', type=str, default="", help="if specified, use instead of getting full model path from name and subdirs")
    
    parser.add_argument('--model_size', default="110m", choices=["110m", "300m", "1_1b"], help="Model size to copy appropriate configs from")
    # TODO, non-uniform weighted averaging, read in data similarity matrix
    
    args = parser.parse_args()
    main(args)
