# OAG-QA-2

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
