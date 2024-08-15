"""WandB metrics."""
import logging
from copy import deepcopy
from typing import Optional

import torch
import numpy as np

from composer.core import State
from composer.core.callback import Callback
from composer.loggers import Logger
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity

log = logging.getLogger(__name__)

__all__ = ['WandBMetrics']


class WandBMetrics(Callback):
    """
        Args:
            num_params_for_flops: total params of network, which can contain partially frozen params
    """
    def __init__(
        self,
        num_params_for_flops: int,
        num_frozen_params: Optional[int] = 0,
        num_eval_params: Optional[int] = 0,
        prev_training_seq_len: Optional[int] = None,
        prev_training_batch_size: Optional[int] = None,
        prev_training_batches: Optional[int] = None,
        prev_training_model_size: Optional[int] = None,
    ):
        # set initial counts for metrics
        if prev_training_seq_len is not None and prev_training_batch_size is not None and prev_training_batches is not None and prev_training_model_size is not None:
            self.initial_compute_tokens_trained_on = prev_training_seq_len * prev_training_batch_size * prev_training_batches
            self.prev_training_model_size = prev_training_model_size / 1e9
        else:
            self.initial_compute_tokens_trained_on = None
            self.prev_training_model_size = None

        # computes GFLOPs
        # assumes self.num_params_for_flops is all of model params
        self.num_frozen_params = num_frozen_params / 1e9
        self.num_eval_params = num_eval_params / 1e9
        self.num_params_for_flops = num_params_for_flops / 1e9


        # these contribute 4ND and not 6ND
        self.num_params_for_flops -= self.num_frozen_params

    def init(self, state: State, logger: Logger) -> None:
        pass

    def _compute_tokens_trained_on(self, state: State):
        current_tokens_trained_on = state.timestamp.token.value

        if self.initial_compute_tokens_trained_on is not None:
            current_tokens_trained_on += self.initial_compute_tokens_trained_on

        return current_tokens_trained_on

    def _compute_gflops(self, state: State):
        current_gflops = 6 * self.num_params_for_flops * state.timestamp.token.value
        current_gflops += 4 * self.num_frozen_params * state.timestamp.token.value
        current_gflops += 2 * self.num_eval_params * state.timestamp.token.value

        if self.initial_compute_tokens_trained_on is not None:
            current_gflops += (6 * self.prev_training_model_size * self.initial_compute_tokens_trained_on)

        return current_gflops

    def _compute_log_gflops(self, state: State):
        current_gflops = 6 * self.num_params_for_flops * state.timestamp.token.value
        current_gflops += 4 * self.num_frozen_params * state.timestamp.token.value
        current_gflops += 2 * self.num_eval_params * state.timestamp.token.value

        if self.initial_compute_tokens_trained_on is not None:
            current_gflops += (6 * self.prev_training_model_size * self.initial_compute_tokens_trained_on)

        return np.log(current_gflops)

    def _compute_metrics(self, metrics):
        computed_metrics = {}
        for metric_name, metric in metrics.items():
            computed_metrics[metric_name] = metric.compute()

        return computed_metrics

    def _compute_log_of_metrics(self, metrics, state: State):
        log_of_metric = {}
        for metric_name, metric in metrics.items():
            # skip plotting log of perplexity (loss)
            if metric_name in 'Perplexity':
                continue

            log_of_metric[f'metrics/{state.dataloader_label}/LOG_' + metric_name] = torch.log(metric)

        return log_of_metric

    def batch_end(self, state: State, logger: Logger) -> None:
        # train metrics have to plotted at batch_end
        metrics = deepcopy(state.train_metrics)

        # these metrics are already plotted within the trainer
        computed_metrics = self._compute_metrics(metrics)
        computed_metrics = self._compute_log_of_metrics(computed_metrics, state)

        # update log_computed_metrics with wall_clock metrics
        computed_metrics.update({'wall_clock/training_tokens': self._compute_tokens_trained_on(state)})
        computed_metrics.update({'wall_clock/GFLOPs': self._compute_gflops(state)})
        computed_metrics.update({'wall_clock/LOG_GFLOPs': self._compute_log_gflops(state)})

        logger.log_metrics(computed_metrics)

    def eval_end(self, state: State, logger: Logger) -> None:
        metrics = {}
        # only apply this callback on LM task metrics during eval
        for metric_name, metric in deepcopy(state.eval_metrics[state.dataloader_label]).items():
            if isinstance(metric, LanguageCrossEntropy) or isinstance(metric, LanguagePerplexity):
                metrics[metric_name] = metric

        if len(metrics) == 0:
            return

        # these metrics are already plotted within the trainer
        computed_metrics = self._compute_metrics(metrics)
        computed_metrics = self._compute_log_of_metrics(computed_metrics, state)
        logger.log_metrics(computed_metrics)
