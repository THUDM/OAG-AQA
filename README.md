# OAG-AQA

## Prerequisites
- Linux
- Python 3.7
- PyTorch 1.10.0+cu111

## Getting Started

### Installation

Clone this repo.


```bash
git clone https://github.com/THUDM/OAG-AQA.git
cd OAG-AQA
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

## OAG-QA Dataset

The raw dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1bFM6QM1tv4cz-Vx8VEGp7A?pwd=v2bb) with password v2bb, [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/AQA/AQA.zip) or [DropBox](https://www.dropbox.com/scl/fi/2ckwl9fcpbik88z1cekot/AQA.zip?rlkey=o7ttmrvpdbvbu3rcr6t33jrx7&dl=1).
The processed data can be downloaded from [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/AQA/aqa_train_data_processed.zip).
Unzip the processed data and put these files into ``data/kddcup/dpr`` directory.

## Run Baseline for [KDD Cup 2024](https://www.biendata.xyz/competition/aqa_kdd_2024/)

We provide a baseline method [DPR](https://arxiv.org/abs/2004.04906).

```bash
cd $project_path
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used
export PYTHONPATH="`pwd`:$PYTHONPATH"
```

### Training DPR
Config the following paths before training (**Absolute** paths are recommended. The same below.)
- ``dpr_stackex_qa`` in _conf/ctx_sources/default_sources.yaml_  
==> ``candidate_papers.tsv`` is descriptions of candidate papers provided in processed data files.
- ``stackex_qa_train`` and ``stackex_qa_valid`` in _conf/datasets/encoder_train_default.yaml_.  
==> ``train_with_hn.json`` and ``dev.json`` are processed training and valiation data provided in processed data files.
- ``pretrained_model_cfg`` and ``pretrained_file`` in _conf/encoder/hf_bert.yaml_.  
==> Download ``bert-base-uncased`` model from [[Hugging Face]](https://huggingface.co/google-bert/bert-base-uncased).

```bash
bash train_dpr.sh
```

### Generating Paper Embeddings
Config the following paths before generating paper embeddings.
- ``model_file`` and ``out_file`` in _conf/gen_embs.yaml_.  
==> ``model_file`` is pre-trained DPR checkpoint. You can use the checkpoint in the last step or use provided checkpoint [[Aliyun Download]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/AQA/aqa_dpr_ckpt.zip).

```bash
python generate_dense_embeddings.py
```

### Retrieval and Evaluation
Config the following paths before retrieval and evaluation.
- ``stackex_qa_test`` in _conf/datasets/retriever_default.yaml_.  
==> ``qa_valid_dpr.tsv`` is the procssed valiation data provided in processed data files.
- ``model_dir`` and ``epoch`` in _dense_retriever.sh_.  
==> ``model_dir`` is the saved model path and ``epoch`` is selcted epoch for evaluation. 


# 1、BM25

Modify `oagqa_data_path,queries_path,output_path` in `tf_idf_stackex.py`

Run `pyhon tf_idf_stackex.py `

# 2、DPR

## requirements

* Python 3.6+
* PyTorch 1.2.0+

## Train

```
CUDA_VISIBLE_DEVICES=2 python train_dense_encoder.py \
    train_datasets=[list of train datasets, comma separated without spaces] \
    dev_datasets=[list of dev datasets, comma separated without spaces] \
    train=biencoder_local \
    output_dir={path to checkpoints dir}
```

The dataset settings can be changed in the file under the path `./DPR/conf/datasets/encoder_train_default.yaml`

For example:

```
CUDA_VISIBLE_DEVICES=2 python train_dense_encoder.py \
    train_datasets=[stackex_qa_train] \
    dev_datasets=[stackex_qa_valid]
    train=biencoder_local \
    output_dir=/workspace/project/DPR/checkpoints/stackex_qa_hn/
```

## Generate embeddings

Modify parameters `model_file, ctx_src, out_file` in the file `./DPR/conf/datasets/encoder_train_default.yaml`

 model_file: A trained bi-encoder checkpoint file to initialize the mode

ctx_src: Name of the all-passages resource

out_file: output .tsv file path to write results to

Run `python generate_dense_embeddings.py`

## Retriever

```
python dense_retriever.py \
	model_file={path to a checkpoint} \
	qa_dataset={the name os the test source} \
	ctx_datatsets=[{list of passage sources's names, comma separated without spaces}] \
	encoded_ctx_files=[{list of encoded document files glob expression, comma separated without spaces}] \
	out_file={path to output json file with results} 
```

# 3、ColBERT

## requirements

* python 3.8+
* pytorch 1.13.1+
* faiss-gpu

## Train

Example usage (training on 4 GPUs):

```
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=4, experiment="nq_bsz32_dim256_lr1e-5")):

        config = ColBERTConfig(
            bsize=32,
            root="/home/shishijie/workspace/project/ColBERT/output",
            doc_maxlen=256,
            dim=256,
            lr=1e-5,
        )
        trainer = Trainer(
            triples="/home/shishijie/workspace/project/oag-qa/data/nq_colbert/triples.tsv",
            queries="/home/shishijie/workspace/project/oag-qa/data/nq_colbert/queries.tsv",
            collection="/home/shishijie/workspace/project/oag-qa/data/nq_colbert/collections.tsv",
            config=config,
        )

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")
```

## Indexing

```
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=4, experiment="nq_bsz32_dim256_lr1e-5")):

        config = ColBERTConfig(
            nbits=2,
            root="/path/to/experiments",
            dim=256,
        )
        indexer = Indexer(checkpoint="{path to checkpoints dir}", config=config)
        indexer.index(name="index_name", collection="path to collections dir")
```

## Retrieval

```
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="nq_bsz32_dim256_lr1e-5")):

        config = ColBERTConfig(
            root="/path/to/experiments",
        )
        searcher = Searcher(index="path to index dir", config=config)
        queries = Queries("path to query file")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("path to save result")
```

# 4、SimLM

## requirement

* python 3.8+
* pytorch 1.13+
* transformers

## Run on NQ datasets

`cd ./SimLM`

### Train a biencoder retriever

```
# Train bi-encoder
bash scripts/dpr/train_nq_biencoder.sh

# Encode corpus passages
bash scripts/dpr/encode_wiki.sh

# Predictions for training datasets can be used as mined hard negatives
bash scripts/dpr/search_dpr.sh
```

### Train a cross-encoder re-ranker

```
# Train cross-encoder re-ranker
bash scripts/dpr/train_nq_reranker.sh

# Re-rank top-100 outputs by biencoder retrievers
bash scripts/dpr/rerank_nq.sh
```

### Train a biencoder retriever with knowledge distillation

```
# Train bi-encoder with knowledge distillation
bash scripts/dpr/train_nq_kd.sh

# Encode corpus passages
bash scripts/dpr/encode_wiki.sh 

# Evaluate on test
bash scripts/dpr/eval_dpr.sh
```
