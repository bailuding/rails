# Retrieval with Learned Similarities (RAILS).
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

# Main entry point to run checkpoint-based evals.

"""
Example usage:

*******************
****** ML-1M ******
*******************
Verified @ 11/18/2024
Config: configs/ml-1m/hstu-mol-sampled-softmax-n128-8x4x64-rails-final.gin
CHECKPOINT=ckpts/ml-1m-l200/HSTU-b8-h2-dqk25-dv25-lsilud0.2-ad0.0_MoL-8x4x64-t0.05-d0.2-l2-q512d0.0swiglu-id0.1-gq128-gi128d0.0-gqi128d0.0-x-glu_silu-uids6040d0.5_local_ssl-n128-lwuid_embedding_l2_norm:0.1-mi_loss:0.001-b128-lr0.001-wu0-wd0.001-2024-11-06_ep72
CUDA_VISIBLE_DEVICES=0 python3 eval_from_checkpoint.py --eval_batch_size=32 --gin_config_file=configs/ml-1m/hstu-mol-sampled-softmax-n128-8x4x64-rails-final.gin  --top_k_method=MoLBruteForceTopK --inference_from_ckpt=$CHECKPOINT
eval @ epoch 72 (189 iters, 6048 evaluated) in 30.00s: NDCG@1 0.0874, NDCG@5 0.1660, NDCG@10 0.1999, NDCG@50 0.2560, NDCG@100 0.2726, HR@1 0.0874, HR@5 0.2422, HR@10 0.3474, HR@50 0.6023, HR@100 0.7041, MRR 0.1694

********************
****** ML-20M ******
********************
Verified @ 11/18/2024
Config: configs/ml-20m/hstu-mol-sampled-softmax-n128-8x4x128-rails-final.gin
CHECKPOINT=ckpts/ml-20m-l200/HSTU-b16-h8-dqk32-dv32-lsilud0.2-ad0.0_MoL-8x4x128-t0.05-d0.2-l2-q512d0.0swiglu-id0.1-gq128-gi128d0.0-gqi128d0.1-x-glu_silu-uids16384d0.8_local_ssl-n128-lwuid_embedding_l2_norm:0.1-mi_loss:0.001-b128-lr0.001-wu0-wd0.001-2024-11-07_ep145
CUDA_VISIBLE_DEVICES=0 python3 eval_from_checkpoint.py --eval_batch_size=32 --gin_config_file=configs/ml-20m/hstu-mol-sampled-softmax-n128-8x4x128-rails-final.gin --top_k_method=MoLBruteForceTopK --inference_from_ckpt=$CHECKPOINT
eval @ epoch 145 (4328 iters, 138496 evaluated) in 119.99s: NDCG@1 0.1024, NDCG@5 0.1879, NDCG@10 0.2202, NDCG@50 0.2772, NDCG@100 0.2933, HR@1 0.1024, HR@5 0.2693, HR@10 0.3695, HR@50 0.6259, HR@100 0.7251, MRR 0.1892

**************************
****** Amazon Books ******
**************************
Verified @ 11/18/2024
Config: configs/amzn-books/hstu-mol-sampled-softmax-n512-8x8x32-rails-final.gin
CHECKPOINT=ckpts/amzn-books-l50/HSTU-b16-h8-dqk8-dv8-lsilud0.5-ad0.0_MoL-8x8x32-t0.05-d0.2-l2-q512d0.0geglu-id0.1-gq128-gi128d0.0-gqi128d0.0-x-glu_silu_local_ssl-n512-lwmi_loss:0.001-ddp2-b64-lr0.001-wu0-wd0.001-2024-11-16-fe5_ep180
CUDA_VISIBLE_DEVICES=0 python3 eval_from_checkpoint.py --limit_eval_to_first_n=8192 --eval_batch_size=32 --gin_config_file=configs/amzn-books/hstu-mol-sampled-softmax-n512-8x8x32-rails-final.gin --top_k_method=MoLBruteForceTopK  --inference_from_ckpt=$CHECKPOINT
eval @ epoch 180 (256 iters, 8192 evaluated) in 69.45s: NDCG@1 0.0178, NDCG@5 0.0322, NDCG@10 0.0386, NDCG@50 0.0539, NDCG@100 0.0604, HR@1 0.0178, HR@5 0.0468, HR@10 0.0668, HR@50 0.1370, HR@100 0.1775, MRR 0.0346
"""

from dataclasses import dataclass
import math, statistics
from typing import Any, Dict, List, Optional, Tuple
import logging
import random
import itertools

from datetime import date
import os
# Hide tensorflow debug messages
# https://stackoverflow.com/questions/65298241/what-does-this-tensorflow-message-mean-any-side-effect-was-the-installation-su
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import time
import gin
import fbgemm_gpu

from absl import app, flags

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from data.reco_dataset import get_reco_dataset
from data.eval import _avg, get_eval_state, eval_metrics_v2_from_tensors
from indexing.utils_rails import get_top_k_module
from modeling.sequential.autoregressive_losses import InBatchNegativesSampler, LocalNegativesSampler
from modeling.sequential.embedding_modules import EmbeddingModule, CategoricalEmbeddingModule, LocalEmbeddingModule
from modeling.sequential.encoder_utils import get_sequential_encoder
from modeling.sequential.input_features_preprocessors import InputFeaturesPreprocessorModule, LearnablePositionalEmbeddingInputFeaturesPreprocessor
from modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor, LayerNormEmbeddingPostprocessor
from modeling.sequential.sasrec import SASRec
from modeling.sequential.hstu import HSTU
from modeling.sequential.features import movielens_seq_features_from_row, SequentialFeatures
from modeling.similarity_utils import get_similarity_function
from trainer.data_loader import create_data_loader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

flags.DEFINE_string(
    "gin_config_file", None, "List of paths to the config file.")

flags.DEFINE_integer("master_port", 12355, "Master port.")
flags.DEFINE_string("inference_from_ckpt", None, "Inference using loaded checkpoint, then exit if set.")
flags.DEFINE_string("top_k_method", None, "Top-K method.")
flags.DEFINE_integer("limit_eval_to_first_n", 0, "Limit eval to first N items.")
flags.DEFINE_integer("eval_batch_size", 64, "Batch size for evals.")
flags.DEFINE_boolean("include_eval_time", False, "Please set this to False for strict accuracy checks.")
flags.DEFINE_string("eval_dtype", "", "If non-empty, run eval in this dtype.")


FLAGS = flags.FLAGS


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True


def setup(rank: int, world_size: int, master_port: int) -> None:
    # Find next available port if the specified one is in use
    current_port = master_port
    while is_port_in_use(current_port):
        current_port += 1
        if current_port > master_port + 20:  # Avoid infinite loop
            raise RuntimeError(f"Could not find available port after trying {master_port} through {current_port-1}")
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(current_port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


@gin.configurable
def train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    inference_from_ckpt: str,
    top_k_method: str,
    limit_eval_to_first_n: int,
    include_eval_time: bool,
    eval_dtype: str,
    dataset_name: str = "ml-20m",
    max_sequence_length: int = 200,
    local_batch_size: int = 128,
    eval_batch_size: int = 128,
    eval_user_max_batch_size: Optional[int] = None,
    main_module: str = "HSTU",
    main_module_bf16: bool = False,
    dropout_rate: float = 0.2,
    linear_hidden_dim: int = 256,
    attention_dim: int = 256,
    user_embedding_norm: str = "l2_norm",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    num_warmup_steps: int = 0,
    sampling_strategy: str = "in-batch",
    num_negatives: int = 1,
    temperature: float = 1.0,
    num_epochs: int = 101,
    eval_interval: int = 1000,
    full_eval_every_n: int = 0,
    loss_module: str = "",
    loss_weights: Dict[str, float] = {},
    item_l2_norm: bool = False,
    save_ckpt_every_n: int = 1000,
    partial_eval_num_iters: int = 32,
    embedding_module_type: str = "local",
    item_embedding_dim: int = 240,
    interaction_module_type: str = "",
    gr_output_length: int = 10,
    mol_bf16_training: bool = False,
    eval_bf16: bool = False,
    l2_norm_eps: float = 1e-6,
    enable_tf32: bool = False,
    opt_enable_apex: bool = False,
) -> None:
    random.seed(42)
    torch.manual_seed(42)

    if eval_dtype == "bf16":
        logging.info("Enabling eval in bf16 to speed up.")

    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32

    logging.info(f"Evaluating model on rank {rank}.")
    setup(rank, world_size, master_port)

    dataset = get_reco_dataset(
        dataset_name=dataset_name,
        max_sequence_length=max_sequence_length,
        chronological=True,
        positional_sampling_ratio=1.0,
    )

    test_data_sampler, test_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=limit_eval_to_first_n == 0,
        drop_last=world_size > 1,
    )

    model_debug_str = main_module
    if embedding_module_type == "categorical":
        embedding_module: EmbeddingModule = CategoricalEmbeddingModule(
            num_items=dataset.max_item_id,
            item_embedding_dim=item_embedding_dim,
            item_id_to_category_id=id_to_category,
        )
    elif embedding_module_type == "local":
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            num_items=dataset.max_item_id,
            item_embedding_dim=item_embedding_dim,
        )
    else:
        raise ValueError(f"Unknown embedding_module_type {embedding_module_type}")
    model_debug_str += f"-{embedding_module.debug_str()}"

    interaction_module, interaction_module_debug_str = get_similarity_function(
        module_type=interaction_module_type,
        query_embedding_dim=item_embedding_dim,
        item_embedding_dim=item_embedding_dim,
    )

    if main_module == "HSTU":
        assert user_embedding_norm == "l2_norm" or user_embedding_norm == "layer_norm", f"Not implemented for {user_embedding_norm}"
        output_postproc_module = (
            L2NormEmbeddingPostprocessor(
                embedding_dim=item_embedding_dim,
                eps=1e-6,
            ) if user_embedding_norm == "l2_norm" else LayerNormEmbeddingPostprocessor(
                embedding_dim=item_embedding_dim,
                eps=1e-6,
            )
        )
        input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=dataset.max_sequence_length + gr_output_length + 1,
            embedding_dim=item_embedding_dim,
            dropout_rate=dropout_rate,
        )
        model = get_sequential_encoder(
            module_type=main_module,
            max_sequence_length=dataset.max_sequence_length,
            max_output_length=gr_output_length + 1,
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
            verbose=True,
        )
        model_debug_str = model.debug_str()
    else:
        raise ValueError(f"Unknown model_debug_str {model_debug_str}.")

    # sampling
    if sampling_strategy == "in-batch":
        negatives_sampler = InBatchNegativesSampler(
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
            dedup_embeddings=True,
        )
        sampling_debug_str = f"in-batch{f'-l2-eps{l2_norm_eps}' if item_l2_norm else ''}-dedup"
    elif sampling_strategy == "local":
        negatives_sampler = LocalNegativesSampler(
            num_items=dataset.max_item_id,
            item_emb=model._embedding_module._item_emb,
            all_item_ids=dataset.all_item_ids,
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
        )
    else:
        raise ValueError(f"Unrecognized sampling strategy {sampling_strategy}.")
    sampling_debug_str = negatives_sampler.debug_str()

    # create model and move it to GPU with id rank
    device = rank
    model = model.to(device)
    if main_module_bf16 or eval_dtype == "bf16":
        model = model.to(torch.bfloat16)
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    date_str = date.today().strftime("%Y-%m-%d")

    def _rename_state_dict(
        state_dict: Dict,
        rename_map: Dict[str, str],
        strict: bool = False,
    ) -> Dict:
        """
        Rename keys in the state dict according to the provided mapping.
        
        Args:
            state_dict: Original state dict
            rename_map: Dictionary mapping old keys to new keys
            strict: If True, raises error when encountering unmapped keys
            
        Returns:
            Dict: New state dict with renamed keys
        """
        new_state_dict = {}
        unmapped_keys = []
        
        for old_key, value in state_dict.items():
            print(f"old_key: {old_key}")
            new_key = old_key
            
            # Check if any pattern in the rename map matches the current key
            for old_pattern, new_pattern in rename_map.items():
                if old_pattern in old_key:
                    new_key = old_key.replace(old_pattern, new_pattern)
                    print(f"Renaming state dict: {old_key} -> {new_key}")
                    break
            
            if old_key == new_key and strict:
                unmapped_keys.append(old_key)
            
            new_state_dict[new_key] = value
            
        if unmapped_keys and strict:
            raise ValueError(
                f"The following keys were not mapped: {unmapped_keys}"
            )
            
        return new_state_dict
    
    checkpoint = torch.load(inference_from_ckpt)
    checkpoint['model_state_dict'] = _rename_state_dict(
        checkpoint['model_state_dict'],
        rename_map={
            "module._ndp_module._item_proj_module.1.weight": "module._ndp_module._item_embeddings_fn._item_emb_proj_module.1.weight",
            "module._ndp_module._item_proj_module.1.bias": "module._ndp_module._item_embeddings_fn._item_emb_proj_module.1.bias",
        },
        strict=False,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    logging.info(f"Restored model and optimizer state from epoch {epoch}'s ckpt: {inference_from_ckpt}. Setting cur_epoch to {epoch}")

    last_training_time = time.time()

    for _ in range(1):
        model.eval()

        # eval per epoch
        eval_dict_all = None
        eval_start_time = time.time()
        float_dtype = torch.bfloat16 if main_module_bf16 or eval_bf16 or eval_dtype == "bf16" else None
        eval_state = get_eval_state(
            model=model.module,
            all_item_ids=dataset.all_item_ids,
            negatives_sampler=negatives_sampler,
            top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                top_k_method=top_k_method,
                model=model.module,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
            ),
            device=device, 
            float_dtype=float_dtype,
        )
        for eval_iter, row in enumerate(iter(test_data_loader)):
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(row, device=device, max_output_length=gr_output_length + 1)
            eval_dict = eval_metrics_v2_from_tensors(
                eval_state, model.module, seq_features, target_ids=target_ids, target_ratings=target_ratings,
                user_max_batch_size=eval_user_max_batch_size,
                include_full_matrices=False,
                include_eval_time=include_eval_time,
                dtype=float_dtype,
            )

            if eval_dict_all is None:
                eval_dict_all = {}
                for k, v in eval_dict.items():
                    eval_dict_all[k] = [v]
            else:
                for k, v in eval_dict.items():
                    eval_dict_all[k] = eval_dict_all[k] + [v]
            del eval_dict

            if limit_eval_to_first_n > 0 and (eval_iter + 1) * eval_batch_size >= limit_eval_to_first_n:
                break

        for k, v in eval_dict_all.items():
            if "eval_time" in k:
                eval_dict_all[k] = list(itertools.chain.from_iterable(v))
            else:
                eval_dict_all[k] = torch.cat(v, dim=-1)
        ndcg_1 = _avg(eval_dict_all["ndcg@1"], world_size=world_size)
        ndcg_5 = _avg(eval_dict_all["ndcg@5"], world_size=world_size)
        ndcg_10 = _avg(eval_dict_all["ndcg@10"], world_size=world_size)
        ndcg_50 = _avg(eval_dict_all["ndcg@50"], world_size=world_size)
        ndcg_100 = _avg(eval_dict_all["ndcg@100"], world_size=world_size)
        hr_1 = _avg(eval_dict_all["hr@1"], world_size=world_size)
        hr_5 = _avg(eval_dict_all["hr@5"], world_size=world_size)
        hr_10 = _avg(eval_dict_all["hr@10"], world_size=world_size)
        hr_50 = _avg(eval_dict_all["hr@50"], world_size=world_size)
        hr_100 = _avg(eval_dict_all["hr@100"], world_size=world_size)
        mrr = _avg(eval_dict_all["mrr"], world_size=world_size)

        logging.info(f"eval @ epoch {epoch} ({eval_iter + 1} iters, {(eval_iter + 1) * eval_batch_size} evaluated) in {time.time() - eval_start_time:.2f}s: "
                     f"NDCG@1 {ndcg_1:.4f}, NDCG@5 {ndcg_5:.4f}, NDCG@10 {ndcg_10:.4f}, NDCG@50 {ndcg_50:.4f}, NDCG@100 {ndcg_100:.4f}, HR@1 {hr_1:.4f}, HR@5 {hr_5:.4f}, HR@10 {hr_10:.4f}, HR@50 {hr_50:.4f}, HR@100 {hr_100:.4f}, MRR {mrr:.4f}")

        if include_eval_time:
            eval_time_avg_ms = 1000 * statistics.mean(eval_dict_all["eval_time"])
            eval_time_dev_ms = 1000 * statistics.stdev(eval_dict_all["eval_time"])
            eval_sample = len(eval_dict_all["eval_time"])
            logging.info(f"EvalTimeAvgMs {eval_time_avg_ms:.2f}, EvalTimeDevMs {eval_time_dev_ms:.2f}, EvalSample { eval_sample }")

        # CSV format
        if include_eval_time:
            logging.info("HR@1,HR@5,HR@10,HR@50,HR@100,BatchTimeMsAvg,BatchTimeMsDev")
            logging.info(f"{hr_1},{hr_5},{hr_10},{hr_50},{hr_100},{eval_time_avg_ms:.3f},{eval_time_dev_ms:.3f}")
        else:
            logging.info("HR@1,HR@5,HR@10,HR@50,HR@100")
            logging.info(f"{hr_1},{hr_5},{hr_10},{hr_50},{hr_100}")

    cleanup()


def mp_train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    gin_config_file: Optional[str],
    inference_from_ckpt: str,
    top_k_method: str,
    limit_eval_to_first_n: int,
    eval_batch_size: int,
    include_eval_time: bool,
    eval_dtype: str,
) -> None:
    if gin_config_file is not None:
        # Hack as absl doesn't support flag parsing inside multiprocessing.
        logging.info(f"Rank {rank}: loading gin config from {gin_config_file}")
        gin.parse_config_file(gin_config_file)

    train_fn(
        rank, world_size, master_port,
        inference_from_ckpt=inference_from_ckpt,
        top_k_method=top_k_method,
        limit_eval_to_first_n=limit_eval_to_first_n,
        eval_batch_size=eval_batch_size,
        eval_user_max_batch_size=eval_batch_size,
        include_eval_time=include_eval_time,
        eval_dtype=eval_dtype,
    )

def main(argv):
    world_size = torch.cuda.device_count()
    assert world_size == 1

    mp.set_start_method('forkserver')
    mp.spawn(mp_train_fn,
             args=(
                world_size,
                FLAGS.master_port,
                FLAGS.gin_config_file,
                FLAGS.inference_from_ckpt,
                FLAGS.top_k_method,
                FLAGS.limit_eval_to_first_n,
                FLAGS.eval_batch_size,
                FLAGS.include_eval_time,
                FLAGS.eval_dtype,
             ),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    app.run(main)
