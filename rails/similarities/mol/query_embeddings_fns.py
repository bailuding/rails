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
Defines functions to generate query-side embeddings for MoL.
"""

import abc
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

from rails.similarities.mol.embeddings_fn import MoLEmbeddingsFn, mask_mixing_weights_fn


def init_mlp_xavier_weights_zero_bias(m) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0.0)


class LMMoLQueryEmbeddingsFn(MoLEmbeddingsFn):
    """
    Generates P_Q query-side embeddings for MoL based on input embeddings and other
    optional tensors for language modeling use cases.
    """

    def __init__(
        self,
        input_max_length: int,
        input_embedding_dim: int,
        dot_product_groups: int,
        dot_product_l2_norm: bool,
        eps: float,
        apply_mixing_weights_v2: bool = False,
        apply_mixing_weights_v4: bool = False,
        mixing_weights_hidden_dim: int = 256,
        filter_invalid_positions: bool = True,
    ) -> None:
        super().__init__()

        self._input_max_length: int = input_max_length
        self._dot_product_groups: int = dot_product_groups
        self._dot_product_l2_norm: bool = dot_product_l2_norm
        self._apply_mixing_weights_v2: bool = apply_mixing_weights_v2
        self._apply_mixing_weights_v4: bool = apply_mixing_weights_v4
        if self._apply_mixing_weights_v2 or self._apply_mixing_weights_v4:
            assert self._apply_mixing_weights_v2 ^ self._apply_mixing_weights_v4
            self._mixing_weights: torch.nn.Module = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=input_embedding_dim,
                    out_features=mixing_weights_hidden_dim,
                ),
                torch.nn.LayerNorm(normalized_shape=[mixing_weights_hidden_dim]),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=mixing_weights_hidden_dim,
                    out_features=self._input_max_length * dot_product_groups,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
        self._filter_invalid_positions: bool = filter_invalid_positions
        self._eps: float = eps

    def forward(
        self,
        input_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (B, N, D,) x float where B is the batch size.

        Returns:
            Tuple of (
                (B, query_dot_product_groups, dot_product_embedding_dim) x float,
                str-keyed aux_losses,
            ).
        """
        if self._apply_mixing_weights_v2 or self._apply_mixing_weights_v4:
            if input_embeddings.size(1) < self._input_max_length:
                input_embeddings = F.pad(
                    input_embeddings,
                    (0, 0, 0, self._input_max_length - input_embeddings.size(1)),
                )
            # eg (256, 8)
            mixing_weights = self._mixing_weights(
                input_embeddings[
                    :,
                    self._dot_product_groups if self._apply_mixing_weights_v4 else 0,
                    :,
                ]
            ).view(
                input_embeddings.size(0),
                self._input_max_length,
                self._dot_product_groups,
            )  # e.g., (B, 256, 8)
            if self._filter_invalid_positions:
                mixing_weights = mask_mixing_weights_fn(
                    mixing_weights,
                    kwargs["input_ids"],
                    input_max_length=self._input_max_length,
                    dot_product_groups=self._dot_product_groups,
                )
            else:
                mixing_weights = F.softmax(mixing_weights, dim=1)
            query_embeddings = torch.einsum(
                "bnd,bnm->bmd", input_embeddings, mixing_weights
            )
        else:
            query_embeddings = input_embeddings[:, : self._dot_product_groups, :]
        if self._dot_product_l2_norm:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        return query_embeddings, {}


class RecoMoLQueryEmbeddingsFn(MoLEmbeddingsFn):
    """
    Generates P_Q query-side embeddings for MoL based on input embeddings and other
    optional tensors for recommendation models. Tested for sequential retrieval
    scenarios.

    The current implementation accesses user_ids associated with the query from
    `user_ids' in kwargs.
    """

    def __init__(
        self,
        query_embedding_dim: int,
        query_dot_product_groups: int,
        dot_product_dimension: int,
        dot_product_l2_norm: bool,
        proj_fn: Callable[[int, int], torch.nn.Module],
        eps: float,
        uid_embedding_hash_sizes: List[int],
        uid_dropout_rate: float,
        uid_embedding_level_dropout: bool = False,
    ) -> None:
        super().__init__()
        self._uid_embedding_hash_sizes: List[int] = uid_embedding_hash_sizes
        self._query_emb_based_dot_product_groups: int = query_dot_product_groups - len(
            self._uid_embedding_hash_sizes
        )
        self._query_emb_proj_module: torch.nn.Module = proj_fn(
            query_embedding_dim,
            dot_product_dimension * self._query_emb_based_dot_product_groups,
        )
        self._dot_product_dimension: int = dot_product_dimension
        self._dot_product_l2_norm: bool = dot_product_l2_norm
        if len(self._uid_embedding_hash_sizes) > 0:
            for i, hash_size in enumerate(self._uid_embedding_hash_sizes):
                setattr(
                    self,
                    f"_uid_embeddings_{i}",
                    torch.nn.Embedding(
                        hash_size + 1, dot_product_dimension, padding_idx=0
                    ),
                )
        self._uid_dropout_rate: float = uid_dropout_rate
        self._uid_embedding_level_dropout: bool = uid_embedding_level_dropout
        self._eps: float = eps

    def forward(
        self,
        input_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (B, query_embedding_dim,) x float where B is the batch size.
            kwargs: str-keyed tensors. Implementation-specific.

        Returns:
            Tuple of (
                (B, query_dot_product_groups, dot_product_embedding_dim) x float,
                str-keyed aux_losses,
            ).
        """
        split_query_embeddings = self._query_emb_proj_module(input_embeddings).reshape(
            (
                input_embeddings.size(0),
                self._query_emb_based_dot_product_groups,
                self._dot_product_dimension,
            )
        )

        aux_losses: Dict[str, torch.Tensor] = {}

        if len(self._uid_embedding_hash_sizes) > 0:
            all_uid_embeddings = []
            for i, hash_size in enumerate(self._uid_embedding_hash_sizes):
                # TODO: decouple this from MoLQueryEmbeddingFn.
                uid_embeddings = getattr(self, f"_uid_embeddings_{i}")(
                    (kwargs["user_ids"] % hash_size) + 1
                )
                if self.training:
                    l2_norm = (uid_embeddings * uid_embeddings).sum(-1).mean()
                    if i == 0:
                        aux_losses["uid_embedding_l2_norm"] = l2_norm
                    else:
                        aux_losses["uid_embedding_l2_norm"] = (
                            aux_losses["uid_embedding_l2_norm"] + l2_norm
                        )

                if self._uid_dropout_rate > 0.0:
                    if self._uid_embedding_level_dropout:
                        # conditionally dropout the entire embedding.
                        if self.training:
                            uid_dropout_mask = (
                                torch.rand(
                                    uid_embeddings.size()[:-1],
                                    device=uid_embeddings.device,
                                )
                                > self._uid_dropout_rate
                            )
                            uid_embeddings = (
                                uid_embeddings
                                * uid_dropout_mask.unsqueeze(-1)
                                / (1.0 - self._uid_dropout_rate)
                            )
                    else:
                        uid_embeddings = F.dropout(
                            uid_embeddings,
                            p=self._uid_dropout_rate,
                            training=self.training,
                        )
                all_uid_embeddings.append(uid_embeddings.unsqueeze(1))
            split_query_embeddings = torch.cat(
                [split_query_embeddings] + all_uid_embeddings, dim=1
            )

        if self._dot_product_l2_norm:
            split_query_embeddings = split_query_embeddings / torch.clamp(
                torch.linalg.norm(
                    split_query_embeddings,
                    ord=None,
                    dim=-1,
                    keepdim=True,
                ),
                min=self._eps,
            )
        return split_query_embeddings, aux_losses
