from typing import Dict, Any, Sequence, Union, Optional, List, Mapping, Tuple
import collections
import gzip
import json
from tqdm import tqdm
import os
from random import Random
import re
import math
import copy
import itertools

from torch.nn.utils.rnn import pad_sequence
from torch import log_softmax
import torch
import transformers
import hf_olmo

from .perplexity_metrics import ppl_metrics, PerplexityMetrics, tensor_args, unsqueeze_args, recursive_tolist

# catwalk and souping utils
DEFAULT_PREDICTION_KWARGS: Dict[str, Any] = {
    "model_max_length": 2048,
    "max_batch_tokens": 20480,
    "batch_size": 32,
    "limit": 1000,
    "split": "validation",
    "random_subsample_seed": 1234,
}

CATWALK_DATA_HOME="/path/to/eval/data/home"
CATWALK_MODELS_HOME="/path/to/hf/models/home"


PPL_METRICS = ppl_metrics()
PPL_METRICS['ppl_metrics'] = PPL_METRICS['ppl_metrics']()

def reset_ppl_metrics():
    global PPL_METRICS
    PPL_METRICS = {'ppl_metrics': PerplexityMetrics()}

    
def get_eval_data(tokenizer, 
                  partition_name: str,
                  data_subdir: Optional[str] = "xppt-s2orc-ppl",
                  split: Optional[str] = "",
                  limit: Optional[int] = DEFAULT_PREDICTION_KWARGS["limit"],
                  random_subsample_seed: Optional[int] = DEFAULT_PREDICTION_KWARGS["random_subsample_seed"], 
                  verbose=False,
                  num_recorded_inputs=3):
    
    doc_instances = get_doc_instances(tokenizer, partition_name, split, limit, random_subsample_seed, data_subdir)

    truncation_length = min(DEFAULT_PREDICTION_KWARGS['model_max_length'], tokenizer.model_max_length)

    ########
    instance_index_to_cc_indices: Mapping[int, List[int]] = collections.defaultdict(list)
    cc_pairs = []

    for instance_index, doc_instance in enumerate(doc_instances):
        rolling_token_windows = list(
            map(
                make_disjoint_window,
                get_rolling_token_windows(
                    token_list=tokenizer.encode(doc_instance, add_special_tokens=False),
                    prefix_token=tokenizer.eos_token_id,
                    max_seq_len=truncation_length,
                    context_len=1,
                ),
            )
        )
        for context, continuation in rolling_token_windows:
            instance_index_to_cc_indices[instance_index].append(len(cc_pairs))
            cc_pairs.append({"input_ids": (
                torch.tensor(context, dtype=torch.long),
                torch.tensor(continuation, dtype=torch.long))})

    # find out the order to process sequences in
    lengths = torch.tensor([
        len(cc_pair["input_ids"][0]) + len(cc_pair["input_ids"][1])
        for cc_pair in cc_pairs
    ], dtype=torch.int)
    ordered_indices = torch.argsort(lengths, descending=True)
    
    return {'doc_instances': doc_instances, 
            'ordered_indices': ordered_indices, 
            'lengths': lengths, 
            'cc_pairs': cc_pairs, 
            'instance_index_to_cc_indices': instance_index_to_cc_indices, 
            'truncation_length': truncation_length}


def combine_datasets(eval_datasets: list):
    combined_dataset = copy.deepcopy(eval_datasets[0])
    for key in ['doc_instances', 'ordered_indices', 'lengths', 'cc_pairs', 'instance_index_to_cc_indices']:
        combined_dataset[key] = list(itertools.chain(*[dataset[key] for dataset in eval_datasets]))
    
    return combined_dataset


def get_doc_instances(tokenizer, 
                  partition_name: str,
                  split: Optional[str] = "",
                  limit: Optional[int] = DEFAULT_PREDICTION_KWARGS["limit"],
                  random_subsample_seed: Optional[int] = DEFAULT_PREDICTION_KWARGS["random_subsample_seed"],
                  data_subdir: Optional[str] = "xppt-s2orc-ppl"):
    instances = get_instances(partition_name, split, limit=limit, random_subsample_seed=random_subsample_seed, data_subdir=data_subdir)
    
    doc_instances: List[str] = [
        instance_as_eleuther_doc(instances[i])
        for i, instance in enumerate(instances)
    ] # not sure if doc part is necessary but oh well
    return doc_instances


def instance_as_eleuther_doc(instance):
    return instance.get('text', instance.get('doc')) # not sure if doc part is necessary but oh well

def get_instances(task: str,
                  split: Optional[str] = "",
                  limit: Optional[int] = DEFAULT_PREDICTION_KWARGS["limit"],
                  random_subsample_seed: Optional[int] = DEFAULT_PREDICTION_KWARGS["random_subsample_seed"],
                  data_subdir: Optional[str] = "xppt-s2orc-ppl") -> Sequence[Dict[str, Any]]:
    instances = get_split(task, split, data_subdir)
    if limit is not None and len(instances) > limit:
        instances = instances[:limit] if random_subsample_seed is None else Random(random_subsample_seed).sample(instances, limit)
    return instances

# for making instances
def get_split(partition_name: str,
              split: str,
              data_subdir: Optional[str] ="xppt-s2orc-ppl",
              extension: str = "jsonl.gz"
             ) -> Sequence[Dict[str, Any]]:
    instances = []
    all_files = []
    # expand directories if need be
    data_dir=os.path.join(CATWALK_DATA_HOME, data_subdir, partition_name)
    if os.path.isdir(data_dir):
        for root, dirs, files in os.walk(data_dir):
            if split in root:
                for file in files:
                    if file.endswith(extension):
                        all_files.append((file, os.path.join(root, file)))
    else:
        all_files.append((partition_name, os.path.join(data_dir)))

    for (orig_file, cache_file) in all_files:
        with gzip.open(cache_file, 'r') as file:
            for line in file:
                instance = json.loads(line.decode("utf-8").strip())
                instance["orig_file_name"] = orig_file
                instances.append(instance)
    return instances

################

def get_model(full_model_path: str):
    model = transformers.models.auto.AutoModelForCausalLM.from_pretrained(full_model_path)
    return model


def get_tokenizer(full_model_path: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(full_model_path)
    return tokenizer


def get_full_model_path(model_name: str, 
                        catwalk_model_subdirs: Union[Optional[str], List[Optional[str]]] = ""):
    if isinstance(catwalk_model_subdirs, str):
        catwalk_model_subdirs = [catwalk_model_subdirs]
    full_model_path = os.path.join(CATWALK_MODELS_HOME, *catwalk_model_subdirs, model_name)
    return full_model_path


################

def get_ppl_scores(model, tokenizer, doc_instances, ordered_indices, lengths, cc_pairs, instance_index_to_cc_indices, truncation_length,
                   batch_size: Optional[int] = DEFAULT_PREDICTION_KWARGS['batch_size'],
                   max_batch_tokens: Optional[int] = DEFAULT_PREDICTION_KWARGS['max_batch_tokens'],
                   verbose: Optional[bool] = False,
                   num_recorded_inputs: Optional[int] = 3):
    log_likelihood_results = run_log_likelihood(
        model, tokenizer,
        doc_instances, ordered_indices, lengths, cc_pairs, instance_index_to_cc_indices, truncation_length,
        batch_size,
        max_batch_tokens, verbose, num_recorded_inputs)
    
    ppl_metric_results = calculate_metrics(log_likelihood_results)
    return ppl_metric_results, log_likelihood_results


def calculate_metrics(predictions: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    metrics = PPL_METRICS
    with tqdm(predictions) as predictions_tqdm:
        for prediction in predictions_tqdm:
            # For models proving model_output (LM models), the metric is called directly
            if 'model_output' in prediction:
                prediction['metrics'] = prediction.get("metrics", {})
                for metric_name, metric in metrics.items():
                    # We'll update the prediction with its individual metrics if need be
                    try:
                        prediction['metrics'].update(metric.get_metrics(prediction))
                    except:
                        # TODO Fix this when needed
                        raise ValueError(f"Metric {metric_name} doesn't support get_metrics")
                    metric.update(prediction)
            else:
                for metric_name, metric_args in prediction.items():
                    try:
                        metric = metrics[metric_name]
                    except KeyError:
                        continue
                    metric_args = tensor_args(metric_args)
                    metric_args = unsqueeze_args(metric_args)
                    metric.update(*metric_args)
    return {
        metric_name: recursive_tolist(metric.compute())
        for metric_name, metric in metrics.items()
    }


def run_log_likelihood(model, tokenizer, doc_instances, ordered_indices, lengths, cc_pairs, instance_index_to_cc_indices, truncation_length,
                       batch_size: Optional[int] = DEFAULT_PREDICTION_KWARGS['batch_size'],
                       max_batch_tokens: Optional[int] = DEFAULT_PREDICTION_KWARGS['max_batch_tokens'],
                       verbose: Optional[bool] = False,
                       num_recorded_inputs: Optional[int] = 3):
    # actually do the processing
    results: List[Optional[float]] = [None] * len(ordered_indices)
    last_index = 0
    
    with torch.inference_mode():
        with tqdm(ordered_indices) as ordered_indices_tqdm:
            while last_index < len(ordered_indices):
                next_batch_size = batch_size
                if max_batch_tokens:
                    current_length = min(lengths[ordered_indices[last_index]].item(), truncation_length)
                    next_batch_size = min(next_batch_size, max_batch_tokens // current_length)
                    if next_batch_size < 1:
                        next_batch_size = 1
                first_index = last_index
                last_index = min(first_index + next_batch_size, len(ordered_indices))
                ordered_indices_tqdm.update(last_index - first_index)
                unpadded_batch = collections.defaultdict(list)
                input_lengths = []
                batch_contexts = []
                batch_continuations = []
                batch_of_indices = ordered_indices[first_index:last_index]
                for index in batch_of_indices:
                    for field_name, (context_ids, continuation_ids) in cc_pairs[index].items():
                        ids = torch.cat([context_ids, continuation_ids])
                        # Use truncation_length+1 since the last token is not in the input
                        if len(ids) > (truncation_length+1):
                            ids = ids[-(truncation_length+1):]
                        ids = ids[:-1]
                        unpadded_batch[field_name].append(ids)

                    input_lengths.append(len(unpadded_batch["input_ids"][-1]))
                    batch_contexts.append(cc_pairs[index]["input_ids"][0])
                    batch_continuations.append(cc_pairs[index]["input_ids"][1])

                padded_batch = {
                    field_name: pad_sequence(tensors, batch_first=True).to(model.device)
                    for field_name, tensors in unpadded_batch.items()
                }

                batch_logits = log_softmax(model(**padded_batch)[0], dim=-1)
                z = zip(batch_of_indices, batch_logits, input_lengths, batch_contexts, batch_continuations)
                for i, instance_logits, input_length, instance_context, instance_continuation in z:
                    instance_logits = instance_logits[input_length-len(instance_continuation):input_length]
                    instance_logits = torch.gather(instance_logits, 1, instance_continuation.unsqueeze(-1).to(model.device))
                    greedy_tokens = instance_logits.argmax(dim=-1)
                    is_greedy = bool((greedy_tokens == instance_continuation.unsqueeze(0).to(model.device)).all())
                    results[i] = {"sum_logits": float(instance_logits.sum()), "num_tokens": len(instance_continuation),
                                 "num_tokens_all": input_length + 1, "is_greedy": is_greedy}
                    if verbose:
                        instance_tokens = [tokenizer.decode(x) for x in instance_continuation]
                        results[i]['tokens'] = instance_tokens
                        results[i]['logits'] = instance_logits.squeeze(-1).tolist()
    del lengths
    assert None not in results
    
    # collect the results
    all_res = []
    for instance_index, doc in enumerate(doc_instances):
        cc_indices = instance_index_to_cc_indices[instance_index]
        results_for_instance = [results[i] for i in cc_indices]
        model_output = {"sum_logits": 0, "num_tokens": 0, "num_tokens_all": 0}
        model_output["num_chars"] = len(doc)
        model_output["num_words"] = len(re.split(r"\s+", doc))
        model_output["num_bytes"] = len(doc.encode("utf-8"))
        for result in results_for_instance:
            model_output["sum_logits"] += result["sum_logits"]
            model_output["num_tokens"] += result["num_tokens"]
            model_output["num_tokens_all"] += result["num_tokens_all"]

        res = {"model_output": model_output}
        if instance_index < num_recorded_inputs:
            res["model_input"] = []
            for index in cc_indices:
                inp = [tokenizer.decode(x) for x in cc_pairs[index]['input_ids']]
                res["model_input"].append(inp)
        all_res.append(res)
    return all_res


def make_disjoint_window(pair):
    """Takes output from get_rolling_token_windows and makes the context not overlap with the continuation"""
    a, b = pair
    return a[: len(a) - (len(b) - 1)], b


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len
        
