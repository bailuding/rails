# Efficient Retrieval with Learned Similarities (RAILS)

## Getting started

Install prerequisites
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

Install Python packages
```
pip3 install transformers faiss-gpu accelerate k-means-constrained pandas gin-config tensorboard
pip3 install torch torchvision torchaudio
pip3 install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121/
```

Download Data:
```
mkdir -p tmp/ && python3 preprocess_public_data.py
```

Run evaluation
```
python eval_batch.py
```
