# modified from olmo

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F

# from ..config import PaddingDirection, TrainConfig

__all__ = ["DataCollator", "PaddingDirection"]

from enum import Enum


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


@dataclass
class DataCollator:
    pad_direction: PaddingDirection
    pad_token_id: int
    composer: bool = False

#     @classmethod
#     def from_train_config(cls, config: TrainConfig) -> DataCollator:
#         return cls(pad_direction=config.data.pad_direction, pad_token_id=config.model.pad_token_id)

    def __call__(self, items: Union[List[Dict[str, Any]], List[torch.Tensor]]) -> Dict[str, Any]:
        assert items
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))
        all_input_ids = []
        all_attention_mask = []
        all_attention_bias = []
        all_indices = []
        all_metadata = []
        for x in items:
            input_ids = x["input_ids"] if isinstance(x, dict) else x
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)

            pad_shape = (
                (max_len - len(input_ids), 0)
                if self.pad_direction == PaddingDirection.left
                else (0, max_len - len(input_ids))
            )

            # Pad input IDs.
            all_input_ids.append(
                F.pad(
                    input_ids.to(dtype=torch.long),
                    pad_shape,
                    value=self.pad_token_id,
                )
            )

            # Pad attention mask.
            attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)
                all_attention_mask.append(
                    F.pad(
                        attention_mask.to(dtype=torch.float),
                        pad_shape,
                        value=0.0,
                    )
                )

            # Pad attention bias.
            attention_bias = x.get("attention_bias") if isinstance(x, dict) else None
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                pad_value = False if attention_bias.dtype == torch.bool else float("-inf")
                all_attention_bias.append(
                    F.pad(
                        attention_bias,
                        pad_shape + pad_shape,
                        value=pad_value,
                    )
                )

            # Indices.
            index = x.get("index") if isinstance(x, dict) else None
            if index is not None:
                all_indices.append(torch.tensor(index))

            # Metadata.
            metadata = x.get("metadata") if isinstance(x, dict) else None
            if metadata is not None:
                all_metadata.append(metadata)

        out: Dict[str, Any] = {"input_ids": torch.stack(all_input_ids)}
        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)
        if all_attention_bias:
            out["attention_bias"] = torch.stack(all_attention_bias)
        if all_indices:
            out["index"] = torch.stack(all_indices)
        if all_metadata:
            out["metadata"] = all_metadata
        if not self.composer:
            return out
        else:
            alt_out = []
            for i in range(len(items)):
                alt_out.append({k: out[k][i] for k in out.keys()})
            return alt_out
