# Retrieval with Learned Similarities (RAILS)

Repository for `Retrieval with Learned Similarities` (RAILS), http://arxiv.org/abs/2407.15462.

TL;DR: while modern retrieval systems have moved beyond simple inner products to more expressive learned similarities, we lack efficient ways to perform retrieval with these advanced methods (e.g., late interactions, arbitrary neural networks, learned index structures/generative retrieval, hybrid solutions, etc.). Our work closes this gap by providing a unifying perspective to neural retrieval. We establish [`Mixture-of-Logits (MoL)`](https://dl.acm.org/doi/10.1145/3580305.3599897) as a universal approximator of _all_ similarity functions, demonstrate that MoL's expressiveness can be realized empirically to achieve superior performance on _diverse, heterogeneous_ scenarios, including finetuning language models for question answering and sequential retrieval models in recommendation systems, and propose efficient techniques to retrieve the approximate top-$k$ results using MoL with tight error bounds. Given MoL’s impressive empirical performance gains of 20%-30% across Hit Rate@50-400 over hundreds of millions to billions of items (e.g., [Zhai et al., 2023](https://dl.acm.org/doi/10.1145/3580305.3599897), [Borisyuk et al., 2024](https://dl.acm.org/doi/10.1145/3627673.3680091)) and broad applicability across heterogeneous scenarios, our work provides strong theoretical and practical justifications for migrating web-scale vector databases away from dense retrieval and MIPS to Retrieval with Learned Similarities (RAILS) on GPUs.

## Getting started

Install prerequisites:
```
sudo apt-get install git-lfs python3.10-dev
```

Set up git-lfs BEFORE cloning the repo:
```
git lfs install --skip-repo
```

If the files in ckpts/amzn-books-l50 are still not downloaded after cloning the repo, run the following command:
```
git lfs pull origin
```

Install python packages:
```
pip3 install transformers faiss-gpu accelerate k-means-constrained pandas gin-config tensorboard
pip3 install torch torchvision torchaudio
pip3 install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121/
```

We've also included `requirements.txt` that can be used alternatively.

### (Optional) Download and preprocess data:
This step is optional as we've included preprocessed data under tmp/.
```
mkdir -p tmp/ && python3 preprocess_public_data.py
```

### Run evaluation
```
python3 eval_batch.py
```

### Re-train recommendation baseline models

You can use `configs/ml-1m/hstu-mol-sampled-softmax-n128-8x4x64-rails-final.gin`, `configs/ml-1m/hstu-sampled-softmax-n128-rails-final.gin`, `configs/ml-20m/hstu-mol-sampled-softmax-n128-8x4x128-rails-final.gin`, `configs/ml-20m/hstu-sampled-softmax-n128-rails-final.gin`, `rails/configs/amzn-books/hstu-mol-sampled-softmax-n512-8x8x32-rails-final.gin`, and `configs/amzn-books/hstu-sampled-softmax-n512-rails-final.gin` to reproduce experiments reported in this paper. We've also included pre-trained checkpoints in `ckpts/` to make it easier to reproduce results.

You should be able to reproduce the following results (verified as of 07/22/2024; non-MoL rows are replicated from results shown in [generative-recommenders github](https://github.com/facebookresearch/generative-recommenders) to faciliate references):

**MovieLens-1M (ML-1M)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | ----------------| --------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.2853           | 0.1603          | 0.5474          | 0.2185          | 0.7528          | 0.2498          |
| BERT4Rec      | 0.2843 (-0.4%)   | 0.1537 (-4.1%)  |                 |                 |                 |                 |
| GRU4Rec       | 0.2811 (-1.5%)   | 0.1648 (+2.8%)  |                 |                 |                 |                 |
| HSTU          | 0.3097 (+8.6%)   | 0.1720 (+7.3%)  | 0.5754 (+5.1%)  | 0.2307 (+5.6%)  | 0.7716 (+2.5%)  | 0.2606 (+4.3%)  |
| HSTU-large       | 0.3294 (+15.5%)     | 0.1893 (+18.1%)     | 0.5935 (+8.4%) | 0.2481 (+13.5%) | 0.7839 (+4.1%) | 0.2771 (+10.9%) |
| HSTU-large + MoL | **0.3412 (+19.6%)** | **0.1979 (+23.5%)** | **0.6013 (+9.8%)** | **0.2556 (+17.0%)** | **0.7877 (+4.6%)** | **0.2840 (+13.7%)** |

**MovieLens-20M (ML-20M)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.2889           | 0.1621          | 0.5503          | 0.2199          | 0.7661          | 0.2527          |
| BERT4Rec      | 0.2816 (-2.5%)   | 0.1703 (+5.1%)  |                 |                 |                 |                 |
| GRU4Rec       | 0.2813 (-2.6%)   | 0.1730 (+6.7%)  |                 |                 |                 |                 |
| HSTU          | 0.3273 (+13.3%)  | 0.1895 (+16.9%) | 0.5889 (+7.0%)  | 0.2473 (+12.5%) | 0.7952 (+3.8%)  | 0.2787 (+10.3%) |
| HSTU-large       | 0.3556 (+23.1%)     | 0.2098 (+29.4%)     | 0.6143 (+11.6%)     | 0.2671 (+21.5%)     | 0.8074 (+5.4%)     | 0.2965 (+17.4%) |
| HSTU-large + MoL | **0.3661 (+26.7%)** | **0.2181 (+34.5%)** | **0.6234 (+13.3%)** | **0.2753 (+25.2%)** | **0.8116 (+5.9%)** | **0.3039 (+20.3%)** |

**Amazon Reviews (Books)**:

| Method        | HR@10            | NDCG@10         | HR@50           | NDCG@50         | HR@200          | NDCG@200        |
| ------------- | ---------------- | ----------------|---------------- | --------------- | --------------- | --------------- |
| SASRec        | 0.0306           | 0.0164          | 0.0754          | 0.0260          | 0.1431          | 0.0362          |
| HSTU          | 0.0416 (+36.4%)  | 0.0227 (+39.3%) | 0.0957 (+27.1%) | 0.0344 (+32.3%) | 0.1735 (+21.3%) | 0.0461 (+27.7%) |
| HSTU-large       | 0.0478 (+56.7%)  | 0.0262 (+60.7%)  | 0.1082 (+43.7%) | 0.0393 (+51.2%) | 0.1908 (+33.4%) | 0.0517 (+43.2%) |
| HSTU-large + MoL | **0.0613 (+100.3%)** | **0.0350 (+113.4%)** | **0.1292 (+71.4%)** | **0.0498 (+91.5%)** | **0.2167 (+51.4%)** | **0.0629 (+73.8%)** |

## References

The code in this repository is intended for reproducing results reported in `Retrieval with Learned Similarities` (http://arxiv.org/abs/2407.15462 and WWW'25 Oral). If you find the work or the code useful, please cite
```
@misc{ding2024retrievallearnedsimilarities,
      title={Retrieval with Learned Similarities}, 
      author={Bailu Ding and Jiaqi Zhai},
      year={2024},
      eprint={2407.15462},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.15462}, 
}
```

Mixture-of-Logits (MoL) was initially proposed in [`Revisiting Neural Retrieval on Accelerators` published in KDD'23](https://arxiv.org/abs/2306.04039):
```
@inproceedings{10.1145/3580305.3599897,
      author = {Zhai, Jiaqi and Gong, Zhaojie and Wang, Yueming and Sun, Xiao and Yan, Zheng and Li, Fu and Liu, Xing},
      title = {Revisiting Neural Retrieval on Accelerators},
      year = {2023},
      isbn = {9798400701030},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3580305.3599897},
      doi = {10.1145/3580305.3599897},
      abstract = {Retrieval finds a small number of relevant candidates from a large corpus for information retrieval and recommendation applications. A key component of retrieval is to model (user, item) similarity, which is commonly represented as the dot product of two learned embeddings. This formulation permits efficient inference, commonly known as Maximum Inner Product Search (MIPS). Despite its popularity, dot products cannot capture complex user-item interactions, which are multifaceted and likely high rank. We hence examine non-dot-product retrieval settings on accelerators, and propose mixture of logits (MoL), which models (user, item) similarity as an adaptive composition of elementary similarity functions. This new formulation is expressive, capable of modeling high rank (user, item) interactions, and further generalizes to the long tail. When combined with a hierarchical retrieval strategy, h-indexer, we are able to scale up MoL to 100M corpus on a single GPU with latency comparable to MIPS baselines. On public datasets, our approach leads to uplifts of up to 77.3\% in hit rate (HR). Experiments on a large recommendation surface at Meta showed strong metric gains and reduced popularity bias, validating the proposed approach's performance and improved generalization.},
      booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
      pages = {5520–5531},
      numpages = {12},
      keywords = {candidate generation, hierarchical retrieval, information retrieval, nearest neighbor search, non-mips retrieval, recommender systems},
      location = {Long Beach, CA, USA},
      series = {KDD '23}
}
```
