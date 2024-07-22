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

# Defines various exact- and approximate- Top-K modules for Mixture-of-Logits (MoL).

from typing import Dict, Tuple

import torch
import faiss

from indexing.candidate_index import TopKModule
from modeling.similarity.mol import MoLSimilarity


class MoLTopKModule(TopKModule):

    def __init__(
        self,
        mol_module: MoLSimilarity, 
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        flatten_item_ids_and_embeddings: bool,
        keep_component_level_item_embeddings: bool,
    ) -> None:
        """
        Args:
            mol_module: MoLSimilarity.
            item_embeddings: (1, X, D)
            item_ids: (1, X,)
            mol_item_embeddings: (K_I, X, D')
        """
        super().__init__()

        self._mol_module: MoLSimilarity = mol_module
        self._item_embeddings: torch.Tensor = item_embeddings if not flatten_item_ids_and_embeddings else item_embeddings.squeeze(0)
        
        if keep_component_level_item_embeddings:
            # (X, D) -> (X, K_I, D) -> (K_I, X, D)
            self._mol_item_embeddings: torch.Tensor = (
                mol_module.get_item_component_embeddings(
                    self._item_embeddings.squeeze(0) if not flatten_item_ids_and_embeddings else self._item_embeddings
                ).permute(1, 0, 2)
            )

        self._item_ids: torch.Tensor = item_ids if not flatten_item_ids_and_embeddings else item_ids.squeeze(0)
        
    @property
    def mol_module(self) -> MoLSimilarity:
        return self._mol_module


class MoLBruteForceTopK(MoLTopKModule):

    def __init__(
        self,
        mol_module: MoLSimilarity, 
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> None:
        super().__init__(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            flatten_item_ids_and_embeddings=False,
            keep_component_level_item_embeddings=False,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        aux_payloads: Dict[str, torch.Tensor],
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, X, ...). Implementation-specific.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.

        Returns:
            Tuple of (top_k_scores x float, top_k_ids x int), both of shape (B, K,)
        """
        # (B, X,)
        all_logits, _ = self.mol_module(query_embeddings, self._item_embeddings, item_sideinfo=None, item_ids=None, aux_payloads=aux_payloads)
        top_k_logits, top_k_indices = torch.topk(
            all_logits, dim=1, k=k, sorted=sorted, largest=True,
        )  # (B, k,)
        return top_k_logits, self._item_ids.squeeze(0)[top_k_indices]


class MoLNaiveTopK(MoLTopKModule):
    """
    The naive top 𝐾 algorithm is a greedy based algorithm to retrieve the top 𝐾 items. The key idea
    is to leverage the dot product scores to scope the set of items for calculating the
    learned similarity scores.

    The algorithm works as the follows:
    • Retrieve the top 𝐾 for each set of embeddings by their dot similarity scores
    • Takes the union of the retrieved items as 𝐼
    • Retrieve the dot similarity scores of all the items in 𝐼 for each embedding set
    • Calculate the learned similarity score for all the items in 𝐼 .
    • Returns the top 𝐾 items from 𝐼 with the highest learned similarity score.
    """

    def __init__(
        self,
        mol_module: MoLSimilarity, 
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        k_per_group: int,
        use_faiss: bool = False,
    ) -> None:
        """
        Args:
            mol_module: MoLSimilarity.
            item_embeddings: (1, X, D).
            item_ids: (1, X,).
            k_per_group: int. top-k (k_per_group) per group to take union.
        """
        super().__init__(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            flatten_item_ids_and_embeddings=True,
            keep_component_level_item_embeddings=True,
        )
        K_I, N, D_prime = self._mol_item_embeddings.size()
        self._k_per_group: int = k_per_group
        self._mol_item_embeddings_t: torch.Tensor = self._mol_item_embeddings.reshape(-1, D_prime).transpose(0, 1)  # (D', K_I * N)
        self._use_faiss = use_faiss
        if use_faiss:
            self._gpu_resources = faiss.StandardGpuResources()
            self._gpu_indexes = []
            nlist = 100
            for i in range(K_I):
                mol_item_embeddings_np = self._mol_item_embeddings[i].to(torch.float32).cpu().numpy().astype('float16')
                quantizer = faiss.IndexFlatIP(D_prime)
                index = faiss.IndexIVFFlat(quantizer, D_prime, nlist, faiss.METRIC_INNER_PRODUCT) 
                assert not index.is_trained  # make sure the index is not already trained
                index.train(mol_item_embeddings_np)  # train with the dataset vectors
                assert index.is_trained  # verify the index is now trained
                gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
                gpu_index.add(mol_item_embeddings_np)
                self._gpu_indexes.append(gpu_index)          
    def forward(
        self, 
        query_embeddings: torch.Tensor,
        k: int,
        aux_payloads: Dict[str, torch.Tensor],
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, D) x float.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.
        """
        B, D = query_embeddings.size()
        mol_query_embeddings, _ = self.mol_module.get_query_component_embeddings(query_embeddings, aux_payloads=aux_payloads)  # (B, K_Q, D)
        _, K_Q, _ = mol_query_embeddings.size()
        K_I, N, _ = self._mol_item_embeddings.size()
        all_indices = []
        if self._use_faiss:
            for i in range(K_Q):
                for j in range(K_I):
                    _, ij_indices = self._gpu_indexes[j].search(mol_query_embeddings[:, i, :].cpu().to(torch.float32).numpy(), self._k_per_group)
                    all_indices.append(torch.tensor(ij_indices, dtype=torch.int64, device=query_embeddings.device))				
        else:
            for i in range(K_Q):
                cur_i_sim_values = torch.mm(mol_query_embeddings[:, i, :], self._mol_item_embeddings_t).view(B * K_I, N)
                _, cur_i_top_k_indices = torch.topk(cur_i_sim_values, k=self._k_per_group, dim=1, sorted=False)
                all_indices.append(cur_i_top_k_indices.view(B, K_I * self._k_per_group))

        sorted_all_indices, _ = torch.sort(torch.cat(all_indices, dim=1), dim=1)
        
        # MoL
        k = K_Q * K_I * self._k_per_group
        filtered_item_embeddings = self._item_embeddings[sorted_all_indices.view(-1)].reshape(B, k, D)
        candidate_scores, _ = self.mol_module(query_embeddings, filtered_item_embeddings, item_sideinfo=None, item_ids=None, aux_payloads=aux_payloads)
        # Mask out duplicate elements across multiple top-k groups, given input is sorted.
        candidate_is_valid = torch.cat(
            [
                torch.ones_like(sorted_all_indices[:, 0:1], dtype=torch.bool),
                sorted_all_indices[:, 1:] != sorted_all_indices[:, :-1]
            ], dim=1,
        )
        candidate_scores = torch.where(candidate_is_valid, candidate_scores, -32767.0)
        top_k_logits, top_k_indices = torch.topk(input=candidate_scores, k=k, dim=1, largest=True, sorted=sorted)
        return top_k_logits, self._item_ids[
            torch.gather(sorted_all_indices, dim=1, index=top_k_indices)
        ]


class MoLAvgTopK(MoLTopKModule):
    def __init__(
        self,
        mol_module: MoLSimilarity, 
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        avg_top_k: int,
    ) -> None:
        """
        Args:
            mol_module: MoLSimilarity.
            item_embeddings: (1, N, D).
            mol_item_embeddings: (K_I, N, D').
            avg_top_k: int.
        """
        super().__init__(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            flatten_item_ids_and_embeddings=True,
            keep_component_level_item_embeddings=True,
        )
        P_X, _, D_prime = self._mol_item_embeddings.size()
        self._avg_mol_item_embeddings_t = (self._mol_item_embeddings.sum(0) / P_X).transpose(0, 1)  # (P_X, X, D') -> (X, D') -> (D', X)
        self._avg_top_k: int = avg_top_k

    def forward(
        self, 
        query_embeddings: torch.Tensor,
        k: int,
        aux_payloads: Dict[str, torch.Tensor],
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, D) x float.
            mol_query_embeddings: (K_Q, B, D) x float.
            k: final top-k to pass to MoL.
        """
        B, D = query_embeddings.size()
        mol_query_embeddings, _ = self.mol_module.get_query_component_embeddings(query_embeddings, aux_payloads=aux_payloads)  # (B, P_Q, D_prime)
        _, P_Q, D_prime = mol_query_embeddings.size()
        P_I, N, _ = self._mol_item_embeddings.size()

        avg_sim_values = torch.mm(mol_query_embeddings.sum(1) / P_Q, self._avg_mol_item_embeddings_t)  # (B, D_prime) * (D_prime, X)
        _, avg_sim_top_k_indices = torch.topk(avg_sim_values, k=self._avg_top_k, dim=1)

        # queries averaged results
        avg_filtered_item_embeddings = self._item_embeddings[avg_sim_top_k_indices[:, :]].reshape(B, self._avg_top_k, D)
        candidate_scores, _ = self.mol_module(query_embeddings, avg_filtered_item_embeddings, item_sideinfo=None, item_ids=None, aux_payloads=aux_payloads)
        top_k_logits, top_k_indices = torch.topk(input=candidate_scores, k=min(k, self._avg_top_k), dim=1, largest=True, sorted=sorted)
        top_k_ids = self._item_ids[torch.gather(avg_sim_top_k_indices, dim=1, index=top_k_indices)]
        if k > self._avg_top_k:
            raise ValueError(f"avg_top_k ({self._avg_top_k}) must be larger than k ({k})")
        return top_k_logits, top_k_ids

    def topk_ids(
        self, 
        query_embeddings: torch.Tensor,
        aux_payloads: Dict[str, torch.Tensor],
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, D) x float.
            mol_query_embeddings: (K_Q, B, D) x float.
            k: final top-k to pass to MoL.
            avg_top_k: average top-k by dot product.
        """
        B, D = query_embeddings.size()
        mol_query_embeddings, _ = self.mol_module.get_query_component_embeddings(query_embeddings, aux_payloads=aux_payloads)  # (B, P_Q, D_prime)
        _, P_Q, D_prime = mol_query_embeddings.size()
        P_I, N, _ = self._mol_item_embeddings.size()

        avg_sim_values = torch.mm(mol_query_embeddings.sum(1) / P_Q, self._avg_mol_item_embeddings_t)  # (B, D_prime) * (D_prime, X)
        _, avg_sim_top_k_indices = torch.topk(avg_sim_values, k=self._avg_top_k, dim=1)
        return avg_sim_top_k_indices


class MoLCombTopK(MoLTopKModule):
    def __init__(
        self,
        mol_module: MoLSimilarity, 
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        avg_top_k: int,
        k_per_group: int,
    ) -> None:
        """
        Args:
            mol_module: MoLSimilarity.
            item_embeddings: (1, N, D).
            mol_item_embeddings: (K_I, N, D').
            avg_top_k: int.
        """
        super().__init__(mol_module=mol_module, item_embeddings=item_embeddings, item_ids=item_ids, flatten_item_ids_and_embeddings=True, keep_component_level_item_embeddings=True)
        # Initialization for naive top K
        P_X, _, D_prime = self._mol_item_embeddings.size()
        self._mol_item_embeddings_t: torch.Tensor = self._mol_item_embeddings.reshape(-1, D_prime).transpose(0, 1)  # (D', K_I * N)
        self._k_per_group: int = k_per_group
        self._avg_top_k: int = avg_top_k
        self._avg_top_k_module = MoLAvgTopK(mol_module, item_embeddings, item_ids, avg_top_k)

    def forward(
        self, 
        query_embeddings: torch.Tensor,
        k: int,
        aux_payloads: Dict[str, torch.Tensor],
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, D) x float.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.
        """
        B, D = query_embeddings.size()
        mol_query_embeddings, _ = self.mol_module.get_query_component_embeddings(query_embeddings, aux_payloads=aux_payloads)  # (B, K_Q, D)
        _, P_Q, _ = mol_query_embeddings.size()
        P_X, X, _ = self._mol_item_embeddings.size()
        all_indices = []
        for i in range(P_Q):
            cur_i_sim_values = torch.mm(mol_query_embeddings[:, i, :], self._mol_item_embeddings_t).view(B * P_X, X)
            _, cur_i_top_k_indices = torch.topk(cur_i_sim_values, k=self._k_per_group, dim=1, sorted=False)
            all_indices.append(cur_i_top_k_indices.view(B, P_X * self._k_per_group))

        avg_topk_ids = self._avg_top_k_module.topk_ids(query_embeddings, aux_payloads=aux_payloads, sorted=sorted)
        all_indices.append(avg_topk_ids)

        sorted_all_indices, _ = torch.sort(torch.cat(all_indices, dim=1), dim=1)

        # MoL
        k = P_Q * P_X * self._k_per_group + self._avg_top_k
        filtered_item_embeddings = self._item_embeddings[sorted_all_indices.view(-1)].reshape(B, k, D)
        candidate_scores, _ = self.mol_module(query_embeddings, filtered_item_embeddings, item_sideinfo=None, item_ids=None, aux_payloads=aux_payloads)
        # Mask out duplicate elements across multiple top-k groups, given input is sorted.
        candidate_is_valid = torch.cat(
            [
                torch.ones_like(sorted_all_indices[:, 0:1], dtype=torch.bool),
                sorted_all_indices[:, 1:] != sorted_all_indices[:, :-1]
            ], dim=1,
        )
        candidate_scores = torch.where(candidate_is_valid, candidate_scores, -32767.0)
        top_k_logits, top_k_indices = torch.topk(input=candidate_scores, k=k, dim=1, largest=True, sorted=sorted)
        return top_k_logits, self._item_ids[
            torch.gather(sorted_all_indices, dim=1, index=top_k_indices)
        ]
