# Efficient Retrieval with Learned Similarities (RAILS).
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

# Defines the query-side transformation functions q -> f_p(q)s, used in
# - Revisiting Neural Retrieval on Accelerators (https://arxiv.org/abs/2306.04039, KDD'23)
# - Efficient Retrieval with Learned Similarities (RAILS).

import abc
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F


class MoLQueryEmbeddingsFn(torch.nn.Module):
    """
    Generates P_Q query-side embeddings for MoL based on input embeddings and other optional tensors
    (which are implementation-specific).
    """

    @abc.abstractmethod
    def forward(
        self,
        input_embeddings: torch.Tensor,
        aux_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (B, query_embedding_dim,) x float where B is the batch size.
            aux_payloads: str-keyed tensors. Implementation-specific.

        Returns:
            Tuple of (
                (B, query_dot_product_groups, dot_product_embedding_dim) x float,
                str-keyed aux_losses,
            ).
        """
        pass


class RecoMoLQueryEmbeddingsFn(MoLQueryEmbeddingsFn):
    """
    Generates P_Q query-side embeddings for MoL based on input embeddings and other optional tensors
    (which are implementation-specific).

    The current implementation is for recommendation datasets specifically and may access user_ids
    associated with the query from `user_ids'.
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
        uid_embedding_l2_weight_decay: float,
        uid_embedding_level_dropout: bool = False,
    ) -> None:
        super().__init__()
        self._uid_embedding_hash_sizes: List[int] = uid_embedding_hash_sizes
        self._query_emb_based_dot_product_groups: int = (
            query_dot_product_groups - len(self._uid_embedding_hash_sizes)
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
                    torch.nn.Embedding(hash_size + 1, dot_product_dimension, padding_idx=0),
                )
        self._uid_dropout_rate: float = uid_dropout_rate
        self._uid_embedding_level_dropout: bool = uid_embedding_level_dropout
        self._uid_embedding_l2_weight_decay: float = uid_embedding_l2_weight_decay
        self._eps: float = eps

    def forward(
        self,
        input_embeddings: torch.Tensor,
        aux_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (B, query_embedding_dim,) x float where B is the batch size.
            aux_payloads: str-keyed tensors. Implementation-specific.

        Returns:
            Tuple of (
                (B, query_dot_product_groups, dot_product_embedding_dim) x float,
                str-keyed aux_losses,
            ).
        """
        split_query_embeddings = self._query_emb_proj_module(input_embeddings).reshape(
            (input_embeddings.size(0), self._query_emb_based_dot_product_groups, self._dot_product_dimension)
        )

        aux_losses: Dict[str, torch.Tensor] = {}

        if len(self._uid_embedding_hash_sizes) > 0:
            all_uid_embeddings = []
            for i, hash_size in enumerate(self._uid_embedding_hash_sizes):
                # TODO: decouple this from MoL and move it to EC later.
                uid_embeddings = getattr(self, f"_uid_embeddings_{i}")(
                    (aux_payloads["user_ids"] % hash_size) + 1
                )
                if self.training and self._uid_embedding_l2_weight_decay > 0.0:
                    l2_norm = (uid_embeddings * uid_embeddings).sum(-1).mean()
                    if i == 0:
                        aux_losses["uid_embedding_l2_norm"] = l2_norm.detach().clone()
                        aux_losses["uid_embedding_l2_loss"] = l2_norm * self._uid_embedding_l2_weight_decay
                    else:
                        aux_losses["uid_embedding_l2_norm"] = aux_losses["uid_embedding_l2_norm"] + l2_norm.detach().clone()
                        aux_losses["uid_embedding_l2_loss"] = aux_losses["uid_embedding_l2_loss"] + l2_norm * self._uid_embedding_l2_weight_decay

                if self._uid_dropout_rate > 0.0:
                    if self._uid_embedding_level_dropout:
                        # conditionally dropout the entire embedding.
                        if self.training:
                            uid_dropout_mask = torch.rand(uid_embeddings.size()[:-1], device=uid_embeddings.device) > self._uid_dropout_rate
                            uid_embeddings = uid_embeddings * uid_dropout_mask.unsqueeze(-1) / (1.0 - self._uid_dropout_rate)
                    else:
                        uid_embeddings = F.dropout(uid_embeddings, p=self._uid_dropout_rate, training=self.training)
                all_uid_embeddings.append(uid_embeddings.unsqueeze(1))
            # print(f"split_query_embeddings.size(): {split_query_embeddings.size()}; all_uid_embeddings.size(): {[x.size() for x in all_uid_embeddings]}")
            split_query_embeddings = torch.cat([split_query_embeddings] + all_uid_embeddings, dim=1)

        if self._dot_product_l2_norm:
            split_query_embeddings = split_query_embeddings / torch.clamp(
                torch.linalg.norm(
                    split_query_embeddings, ord=None, dim=-1, keepdim=True,
                ), min=self._eps,
            )
        return split_query_embeddings, aux_losses