# ez-data-ablations
Public repository for "Approximating Training Data Ablations for Language Models through Model Merging".

This is a work in progress. Please feel free to reach out with any questions!

<!-- TOC -->
* [ez-data-ablations](#ez-data-ablations)
* [Setup](#setup)
* [Preparing Data](#preparing-data)
* [Training a Model](#training-a-model)
* [Evaluation](#evaluation)
* [Figures](#figures)
* [Misc.](#misc)
<!-- TOC -->


# Setup

Environment for training:
```shell
conda create -n ablation-sim python=3.10
pip install -r requirements.txt
```

Evaluation has dependencies different from training. 
See [OLMo-Eval](https://github.com/allenai/OLMo-Eval) and [PALOMA evaluation](https://github.com/allenai/OLMo-Eval/blob/main/paloma/README.md) for relevant *evaluation* setup and [Evaluation](#evaluation) for additional details. 

# Preparing Data

As described in our paper, we use English Wikipedia and Gutenberg as seed pre-training data, and we use data from up to two high-level sources ([S2ORC](https://aclanthology.org/2020.acl-main.447/) and [M2D2 Wikipedia](https://aclanthology.org/2022.emnlp-main.63/)) as continued pre-training data.
The data will be available on HuggingFace shortly (TODO 1)

The notebook at ([notebooks](notebooks) TODO 2) documents the procedure used to determine the exact splits.

## Evaluation data

Our training scripts expect tokenized inputs, but evaluation data for offline perplexity evaluations as described in [Evaluation](#evaluation) should remain in `.jsonl`[.gz] format.

# Training a Model

We provide an example script for training a seed model at ([scripts](scripts) TODO 3). 
Seed models are available at (TODO 4). 

We provide scripts corresponding to different experiments in the paper in ([configs](configs) TODO 5, + commands with args)

# Evaluation
Although the training code can include in-loop evaluations, perplexity evaluations were performed _post-training_.

Our codebase is Mosaic ML Composer-like, and the model checkpoints resulting from training need to be **converted** to an OLMo/HF-compatible format (see [xp-pt-as-olmo README](xp-pt-as-olmo/readme.md) for an example usage of the [conversion script](xp-pt-as-olmo/convert_checkpoint.py) before offline evaluation.
See [OLMo-Eval](https://github.com/allenai/OLMo-Eval) and [PALOMA evaluation](https://github.com/allenai/OLMo-Eval/blob/main/paloma/README.md) for relevant setup.

In practice, we largely rely on a lightweight version of the OLMo-Eval framework and write results to simple text files to accommodate on-demand perplexity evaluations which vary from model to model. 
This lightweight eval code is included in this repo, and commands to call the appropriate scripts ([eval_model_data.py](eval_model_data.py) and [eval_model_data-paloma.py](eval_model_data-paloma.py)) can be generated by [quick-eval-commands.py](quick-eval-commands.py). 
See example commands in ([scripts](scripts) TODO 6).

In our work, we find that perplexity evaluations of models trained on a data mixture are highly correlated with perplexity evaluations of merged (parameter averaged) models, 
where the component models in the average are each trained on separate partitions of the data mixture. 
Although the ad-hoc evaluation scripts provided can evaluate parameter averaged models on the fly given individual components,
it may sometimes be helpful to explicitly save a parameter averaged model to disk (e.g. to evaluate a merge of merged models, as in "macro"-merged models). 
See [save_uniform_avg_model.py](save_uniform_avg_model.py).

For the results seen in our paper, we include a notebook ([scripts](scripts) TODO 7) used to compile and analyze results, as well as generate figures


# Figures

See our notebooks for details on figure generation. We include pdf and png versions of our figures for convenience ([figures](figures) TODO 8)


# Misc.

- Preprint/paper citation to come
- Our small models have 130m parameters but are referred to as 110m parameter models throughout this repo

- 

