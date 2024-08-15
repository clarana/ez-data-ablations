"""
A script to convert a xp-pt checkpoint to a xp-pt-as-olmo checkpoint (loadable with HF).
"""

import argparse
import torch
import yaml
import os

def main(args):

    # read yaml config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    d_model =  config["model"]["d_model"]
    n_layers = config["model"]["n_layers"]

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # make new checkpoint
    new_state_dict = {}
    for k in checkpoint['state']['model'].keys():
        new_k = k.replace('model.', 'transformer.')
        new_k = new_k.replace('att_norm.', 'attn_norm.')
        new_k = new_k.replace('ffn_norm.', 'ff_norm.')
        new_state_dict[new_k] = checkpoint['state']['model'][k]

    # make dumby weights and bias for attn_out
    for i in range(n_layers):
        # identity
        new_state_dict[f'transformer.blocks.{i}.attn_out.weight'] = torch.eye(d_model, d_model)
        new_state_dict[f'transformer.blocks.{i}.attn_out.bias'] = torch.zeros(d_model)

    # make output dir if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # save new checkpoint
    torch.save(new_state_dict, os.path.join(args.output_dir, 'model.pt'))

    # copy over config
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    

if __name__ == "__main__":
    #args
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)