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

import abc
from typing import Dict, Optional, Tuple

import torch


class SimilarityModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            query_embeddings: (B, input_embedding_dim) x float.
            item_embeddings: (1/B, X, item_embedding_dim) x float.
            **kwargs: Implementation-specific keys/values (e.g.,
                item sideinfo, etc.)

        Returns:
            A tuple of (
                (B, X,) similarity values,
                keyed outputs representing auxiliary losses at training time.
            ).
        """
        pass
