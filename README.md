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
python -m spacy download en_core_web_sm
```

## OAG-QA Dataset

The raw dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1bFM6QM1tv4cz-Vx8VEGp7A?pwd=v2bb) with password v2bb, [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/AQA/AQA.zip) or [DropBox](https://www.dropbox.com/scl/fi/2ckwl9fcpbik88z1cekot/AQA.zip?rlkey=o7ttmrvpdbvbu3rcr6t33jrx7&dl=1).
The processed data can be downloaded from [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/AQA/aqa_train_data_processed.zip).
Unzip the processed data and put these files into ``data/kddcup/dpr`` directory.  

Note: In train_with_hn.json, ``negative_ctxs`` are randomly sampled from candidate papers, and ``hard_negative_ctxs`` are randomly sampled from the references of positive samples.  References of positive samples are sampled from [[DBLP Citation Dataset]](https://open.aminer.cn/open/article?id=655db2202ab17a072284bc0c).

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
==> Download ``bert-base-uncased`` model from [[Aliyun]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/AQA/bert-base-uncased-dpr-init.zip).

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

```bash
bash dense_retriever.sh
```

### Output
The output files for valiation submission is in the same directory as ``model_dir``. 
We evaluate the checkpoint at epoch 29, and the MAP value on validation set is 0.16909.

## Citation

If you find this dataset useful in your research, please cite the following papers:

```
@inproceedings{tam2023parameter,
  title={Parameter-Efficient Prompt Tuning Makes Generalized and Calibrated Neural Text Retrievers},
  author={Weng Tam and Xiao Liu and Kaixuan Ji and Lilong Xue and Jiahua Liu and Tao Li and Yuxiao Dong and Jie Tang},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={13117--13130},
  year={2023}
}

@article{zhang2024oag,
    title={OAG-Bench: A Human-Curated Benchmark for Academic Graph Mining},
    author={Fanjin Zhang and Shijie Shi and Yifan Zhu and Bo Chen and Yukuo Cen and Jifan Yu and Yelin Chen and Lulu Wang and Qingfei Zhao and Yuqing Cheng and Tianyi Han and Yuwei An and Dan Zhang and Weng Lam Tam and Kun Cao and Yunhe Pang and Xinyu Guan and Huihui Yuan and Jian Song and Xiaoyan Li and Yuxiao Dong and Jie Tang},
    journal={arXiv preprint arXiv:2402.15810},
    year={2024}
}
```
