# Retrieval with Learned Similarities (RAILS, https://arxiv.org/abs/2407.15462).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Defines interface for generating query- and item-side embeddings for MoL.
"""

import abc
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F


def _mask_mixing_weights_fn_static_shape(
    mixing_weights: torch.Tensor,
    input_ids: torch.Tensor,
    dot_product_groups: int,
) -> torch.Tensor:
    valid_positions = (
        (input_ids != 0).unsqueeze(2).expand(-1, -1, dot_product_groups)
    )  # [bs,seq_len, dot_product_groups]
    mixing_weights = torch.where(
        valid_positions, mixing_weights, torch.ones_like(mixing_weights) * -1e3
    )
    return F.softmax(mixing_weights, dim=1)


@torch.compile(dynamic=True)
def mask_mixing_weights_fn(
    mixing_weights: torch.Tensor,
    input_ids: torch.Tensor,
    input_max_length: int,
    dot_product_groups: int,
) -> torch.Tensor:
    if input_ids.size(1) < input_max_length:
        input_ids = F.pad(input_ids, (0, input_max_length - input_ids.size(1)))
    return _mask_mixing_weights_fn_static_shape(
        mixing_weights=mixing_weights,
        input_ids=input_ids,
        dot_product_groups=dot_product_groups,
    )


class MoLEmbeddingsFn(torch.nn.Module):
    """
    Generates K_Q query-side (K_I item-side) embeddings for MoL based on
    input embeddings and other optional implementation-specific tensors.
    """

    @abc.abstractmethod
    def forward(
        self,
        input_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (B, ...) x float where B is the batch size.

        Returns:
            Tuple of (
                (B, query_dot_product_groups/item_dot_product_groups, dot_product_embedding_dim) x float,
                str-keyed auxiliary losses.
            ).
        """
        pass
