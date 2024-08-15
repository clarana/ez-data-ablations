"""
Adapted from
[MosaicML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git) and
Pete's code for custom LLM at AI2
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import NamedTuple, Optional, cast

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from composer.models import ComposerModel
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity

from evaluator.metrics import ICLMetric

from torchmetrics import Metric


__all__ = [
    "RMSLayerNorm",
    "LayerNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "CustomGPTBlock",
    "CustomGPT",
]


CUSTOMGPT_CONFIGS = {
    'customgpt-110m': {
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'mlp_ratio': 4,
        'attention_dropout': 0.0,
        'attention_layer_norm': True,
        'layer_norm_type': 'low_precision',   # if not compiling, use 'low_precision'
        'residual_dropout': 0.0,
        'embedding_dropout': 0.0,
        'max_sequence_length': 2048,
        'vocab_size': 50280,
        'embedding_size': 50304,    # number of embeddings in the table, different than vocab_size (multiple of 128)
        'bos_token_id': None,       # assigned by tokenizer
        'eos_token_id': None,       # assigned by tokenizer
        'pad_token_id': None,       # assigned by tokenizer
        'init_std': 0.02,           # check init for opt/bloom
    },
    'customgpt-300m': {
        'd_model': 1024,
        'n_heads': 16,
        'n_layers': 24,
        'mlp_ratio': 4,
        'attention_dropout': 0.0,
        'attention_layer_norm': True,
        'layer_norm_type': 'low_precision',   # if not compiling, use 'low_precision'
        'residual_dropout': 0.0,
        'embedding_dropout': 0.0,
        'max_sequence_length': 2048,
        'vocab_size': 50280,
        'embedding_size': 50304,    # number of embeddings in the table, different than vocab_size (multiple of 128)
        'bos_token_id': None,       # assigned by tokenizer
        'eos_token_id': None,       # assigned by tokenizer
        'pad_token_id': None,       # assigned by tokenizer
        'init_std': 0.02,           # check init for opt/bloom
    },
    'customgpt-1.1b': {
        'd_model': 2048,
        'n_heads': 16,
        'n_layers': 24,
        'mlp_ratio': 4,
        'attention_dropout': 0.0,
        'attention_layer_norm': True,
        'layer_norm_type': 'low_precision',   # if not compiling, use 'low_precision'
        'residual_dropout': 0.0,
        'embedding_dropout': 0.0,
        'max_sequence_length': 2048,
        'vocab_size': 50280,
        'embedding_size': 50304,    # number of embeddings in the table, different than vocab_size (multiple of 128)
        'bos_token_id': None,       # assigned by tokenizer
        'eos_token_id': None,       # assigned by tokenizer
        'pad_token_id': None,       # assigned by tokenizer
        'init_std': 0.02,           # check init for opt/bloom
    },
}


class LayerNorm(nn.Module):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.normalized_shape = (config['d_model'],)
        self.eps = 1e-05
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.low_precision = True if self.config['layer_norm_type'] == 'low_precision' else False

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            if tensor.device.type == "cuda":
                dtype = torch.get_autocast_gpu_dtype()
            elif tensor.device.type == "cpu":
                dtype = torch.get_autocast_cpu_dtype()
            else:
                raise NotImplementedError()
            return tensor.to(dtype=dtype)
        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class RMSLayerNorm(nn.Module):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation that can optionally run
    in low-precision.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.eps = 1e-08
        self.weight = nn.Parameter(torch.ones(self.config['d_model']))
        self.bias = nn.Parameter(torch.zeros(self.config['d_model']))
        self.low_precision = True if self.config['layer_norm_type'] == 'low_precision' else False

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            if tensor.device.type == "cuda":
                dtype = torch.get_autocast_gpu_dtype()
            elif tensor.device.type == "cpu":
                dtype = torch.get_autocast_cpu_dtype()
            else:
                raise NotImplementedError()
            return tensor.to(dtype=dtype)
        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = self._cast_if_autocast_enabled(self.weight)
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return self.rms_norm(downcast_x, downcast_weight, downcast_bias)
        else:
            return self.rms_norm(x, self.weight, self.bias)

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)

        rms_x = norm_x * self.config['d_model'] ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if bias is not None:
            return weight * x_normed + self.bias
        else:
            return weight * x_normed


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: dict):
        super().__init__()

        dim = config['d_model'] // config['n_heads']
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len):
        seq = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)  # type: ignore
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    B, nh, T, hs = x.size()
    x = x.view(B, nh, T, 2, hs // 2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    out = (t * pos.cos()) + (rotate_half(t) * pos.sin())
    return out.to(t.dtype)


class SwiGLU(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


class CustomGPTBlock(nn.Module):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x)) + Attention(LN(x))``
    as in the PaLM architecture, as opposed to the typical ``MLP(LN(x + Attention(LN(x))))``.
    The decoupling of the MLP and Attention functions allow us to fuse the separate input projections
    into a single linear layer to increase throughput. In this configuration it's also straight-forward
    to fuse the output projections, but we found that didn't help.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        assert config['d_model'] % config['n_heads'] == 0

        # Dropout.
        self.dropout = nn.Dropout(config['residual_dropout'])

        # Layer norms.
        self.att_norm = LayerNorm(config)
        self.ffn_norm = LayerNorm(config)

        self.k_norm: Optional[LayerNorm] = None
        self.q_norm: Optional[LayerNorm] = None

        if config['attention_layer_norm']:
            self.k_norm = LayerNorm(config)
            self.q_norm = LayerNorm(config)

        # Activation function.
        self.act = SwiGLU(config)
        assert (self.act.output_multiplier * config['mlp_ratio'] * config['d_model']) % 1 == 0

        # Fused attention projection
        self.fused_dims = (config['d_model'], config['d_model'], config['d_model'])
        self.att_proj = nn.Linear(config['d_model'], sum(self.fused_dims))
        self.att_proj._fused = (0, self.fused_dims)  # type: ignore

        # FFN module
        self.ff_proj = nn.Linear(config['d_model'], config['mlp_ratio'] * config['d_model'])
        self.ff_out = nn.Linear(int(self.act.output_multiplier * config['mlp_ratio'] * config['d_model']), config['d_model'])
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        self.rotary_emb = RotaryEmbedding(config)
        self.register_buffer("pos_emb", self.rotary_emb(config['max_sequence_length']), persistent=False)

    def get_rotary_embedding(self, seq_len: int) -> torch.Tensor:
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= seq_len:  # type: ignore
            return self.pos_emb[:seq_len]  # type: ignore

        pos_emb = self.rotary_emb(seq_len)
        self.register_buffer("pos_emb", pos_emb, persistent=False)

        return pos_emb

    def attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_bias: Optional[torch.FloatTensor] = None
    ) -> torch.Tensor:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape (all): (B, nh, T, hs)
        q = q.view(B, T, self.config['n_heads'], C // self.config['n_heads']).transpose(1, 2)
        k = k.view(B, T, self.config['n_heads'], C // self.config['n_heads']).transpose(1, 2)
        v = v.view(B, T, self.config['n_heads'], C // self.config['n_heads']).transpose(1, 2)

        # Apply rotary embeddings.
        positions = self.get_rotary_embedding(T)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None if attention_bias is None else attention_bias.to(dtype=dtype),
            dropout_p=0.0 if not self.training else self.config['attention_dropout'],
            is_causal=attention_bias is None,
        )

        # Re-assemble all head outputs side-by-side.
        # shape: (B, T, C)
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        return att

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        q, k, v = self.att_proj(self.att_norm(x)).split(self.fused_dims, dim=-1)

        # Get attention scores.
        # shape: (B, T, C)
        att = self.attention(q, k, v, attention_bias)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        x = x + self.dropout(self.ff_out(self.act(self.ff_proj(self.ffn_norm(x)))))

        return x


class CustomGPT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        # enable flash attention in scaled_dot_product_attention
        torch.backends.cuda.enable_flash_sdp(True)

        if self.config['embedding_size'] is not None and self.config['embedding_size'] != self.config['vocab_size']:
            if self.config['embedding_size'] < self.config['vocab_size']:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config['embedding_size'] % 128 != 0:
                import warnings

                warnings.warn("Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning)

        # transformer params
        self.wte = nn.Embedding(config['embedding_size'] or config['vocab_size'], config['d_model'])
        self.emb_drop = nn.Dropout(config['embedding_dropout'])
        self.blocks = nn.ModuleList([CustomGPTBlock(config) for _ in range(config['n_layers'])])
        self.ln_f = LayerNorm(config)

        # initialize
        self.apply(self.param_init_fn)

    def fsdp_wrap_fn(self, module):
        return isinstance(module, CustomGPTBlock)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, CustomGPTBlock)

    @property
    def causal_attention_bias(self) -> torch.FloatTensor:
        if not hasattr(self, "_causal_attention_bias"):
            att_bias = torch.triu(
                torch.ones(self.config['max_sequence_length'], self.config['max_sequence_length'], dtype=torch.float),
                diagonal=1,
            )
            att_bias.masked_fill_(att_bias == 1, float("-inf"))
            self.register_buffer(
                "_causal_attention_bias",
                att_bias.to(dtype=self.buffer_dtype).view(1, 1, self.config['max_sequence_length'], self.config['max_sequence_length']),
                persistent=False,
            )

        return self._causal_attention_bias  # type: ignore[return-type]

    def param_init_fn(self, module):
        from functools import partial

        init_fn = partial(nn.init.normal_, mean=0.0, std=self.config['init_std'])

        def fused_init_fn(module):
            # Parameter initialization is often based on the parameters shape.
            # If a layer is fused, initialization should be based on the shapes
            # of the original tensor instead of the shape of the fused tensor.
            # Layers which are fused should have the _fused attribute defined.
            # The first element of _fused is the dimension along which the tensor is fused.
            # This is followed by an iterable of split indices.
            _fused = getattr(module, "_fused", None)
            if _fused is None:
                raise RuntimeError("Internal logic error")

            dim, splits = _fused
            splits = (0, *splits, module.weight.size(dim))
            for s, e in zip(splits[:-1], splits[1:]):
                slice_indices = [slice(None)] * module.weight.ndim
                slice_indices[dim] = slice(s, e)
                init_fn(module.weight[slice_indices])

        # Linear
        if isinstance(module, nn.Linear):
            if hasattr(module, "_fused"):
                fused_init_fn(module)
            else:
                init_fn(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

            if getattr(module, "_is_residual", False):
                with torch.no_grad():
                    module.weight.div_(math.sqrt(2 * self.config['n_layers']))

        # Embedding
        if isinstance(module, nn.Embedding):
            init_fn(module.weight)

        # LayerNorm
        if isinstance(module, (nn.LayerNorm, RMSLayerNorm, LayerNorm)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        batch_size, seq_len = input_ids.size()
        assert seq_len <= self.config['max_sequence_length'], (
            f"Cannot forward input with seq_len={seq_len}, "
            f"this model only supports seq_len<={self.config['max_sequence_length']}"
        )

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.wte(input_ids)  # type: ignore

        # apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=x.dtype).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
            attention_mask.masked_fill_(attention_mask == 1.0, float("-inf"))

        # Merge attention mask with attention bias.
        if attention_bias is not None or attention_mask is not None:
            if attention_bias is None:
                # Default to causal attention bias.
                attention_bias = self.causal_attention_bias
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=x.dtype)
                attention_bias.masked_fill_(attention_bias == 0.0, float("-inf"))

            attention_bias = attention_bias[:, :, :seq_len, :seq_len]

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask

        # Apply blocks one-by-one.
        for block in self.blocks:  # type: ignore
            x = block(x, attention_bias=attention_bias)

        # Apply final layer norm.
        # shape: (batch_size, seq_len, d_model)
        x = self.ln_f(x)  # type: ignore

        # Get logits.
        # shape: (batch_size, seq_len, vocab_size)
        logits = F.linear(x, self.wte.weight, None)  # type: ignore

        return logits


class TrainBatchPerplexity(Metric):
    """
    A metric for tracking training perplexity on a per-batch basis.
    We use this as a training metric instead of composer's built-in
    :class:`LanguageCrossEntropy` to avoid recomputing the loss.
    """

    def __init__(self) -> None:
        super().__init__(sync_on_compute=False)
        self.loss: Optional[torch.Tensor]

    def update(self, loss: torch.Tensor):
        self.loss = loss

    def compute(self) -> torch.Tensor:
        assert self.loss is not None
        return torch.exp(self.loss)


class ComposerCustomGPT(ComposerModel):
    def __init__(self, config):
        super().__init__()

        self.model = CustomGPT(config)

        # param counts, no additional lm_head weights => no double counting
        self.num_embedding_params = sum(param.numel() for param in self.model.wte.parameters())
        self.num_params_for_flops = sum(param.numel() for param in self.model.parameters())

        print('___________________________________________________________________________')
        print(f'Number of embedding params: {self.num_embedding_params / 1e6:.2f}M')
        print(f'Param count for FLOPs computation: {self.num_params_for_flops / 1e6:.2f}M')
        print('___________________________________________________________________________')

        self.train_metrics = {
            "Perplexity": TrainBatchPerplexity(),
        }

        self.eval_metrics = {
            "Perplexity": LanguagePerplexity(),
            "CrossEntropy": LanguageCrossEntropy(),
            "acc": ICLMetric(),
            "f1": ICLMetric(metric_type='f1'),
            "len_norm": ICLMetric(metric_type='len_norm'),
            "pmi_dc": ICLMetric(metric_type='pmi_dc'),
        }

    def get_labels(self, batch):
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, attention_mask = batch["input_ids"], batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0.0, -100)

        return labels

    def forward(self, batch):
        # make sure batch doesn't contain 'attention_mask' to run flash attention
        if 'attention_mask' in batch:
            del batch['attention_mask']

        logits = self.model(**batch)
        labels = self.get_labels(batch)

        # remove last element from each sequence in logits
        # remove 0th element from each sequence in labels
        loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)), 
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100
        )

        # domain conditional input
        if 'dc_input_ids' in batch:
            dc_logits = self.model(input_ids=batch['dc_input_ids'])

            return {"logits": logits, "dc_logits": dc_logits, "labels": labels, "loss": loss}

        return {"logits": logits, "labels": labels, "loss": loss}

    def loss(self, outputs, batch):
        del batch
        return outputs["loss"]

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric) -> None:
        if isinstance(metric, ICLMetric) and 'ctx' in batch.keys() and 'continuation' in batch.keys():
            if 'dc_input_ids' in batch:
                metric.update(batch, outputs["logits"], outputs["dc_logits"])
            else:
                metric.update(batch, outputs["logits"])
        else:
            del batch
            if isinstance(metric, TrainBatchPerplexity):
                metric.update(outputs["loss"].detach())
            elif isinstance(metric, LanguagePerplexity) or isinstance(metric, LanguageCrossEntropy):
                # logits and labels are not aligned in this case
                logits, labels = outputs["logits"], outputs["labels"]
                metric.update(logits[..., :-1, :].contiguous().view(-1, logits.size(-1)), labels[..., 1:].contiguous().view(-1))


def create_customgpt(
    model_name: str,
    tokenizer: AutoTokenizer,
    model_path: Optional[str] = None,
    model_config: Optional[dict] = None,
):
    if not model_config:
        model_config = {}

    # gather param count for FLOP computation from model being loaded
    prev_params_for_flops = -1

    config = CUSTOMGPT_CONFIGS[model_name]

    # set config token ids from tokenizer
    config['bos_token_id'] = tokenizer.bos_token_id
    config['eos_token_id'] = tokenizer.eos_token_id
    config['pad_token_ids'] = tokenizer.pad_token_id

    model = ComposerCustomGPT(config)

    if model_path is not None:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state']['model']
        model.load_state_dict(state_dict, strict=True)

        # CustomGPT doesn't store separate lm_head weights inside state_dict
        prev_params_for_flops = sum(param.numel() for _, param in state_dict.items())

    return model, prev_params_for_flops
