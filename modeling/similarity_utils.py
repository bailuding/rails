# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from typing import Tuple

import gin
import torch

from modeling.similarity.dot_product import DotProductSimilarity
from modeling.similarity.mol_utils import create_mol_interaction_module


@gin.configurable
def get_similarity_function(
    module_type: str,
    query_embedding_dim: int,
    item_embedding_dim: int,
    bf16_training: bool = False,
    activation_checkpoint: bool = False,
) -> Tuple[torch.nn.Module, str]:
    if module_type == "DotProduct":
        interaction_module = DotProductSimilarity()
        interaction_module_debug_str = "DotProduct"
    elif module_type == "MoL":
        interaction_module, interaction_module_debug_str = create_mol_interaction_module(
            query_embedding_dim=query_embedding_dim,
            item_embedding_dim=item_embedding_dim,
            bf16_training=bf16_training,
        )
    else:
        raise ValueError(f"Unknown interaction_module_type {module_type}")
    return interaction_module, interaction_module_debug_str