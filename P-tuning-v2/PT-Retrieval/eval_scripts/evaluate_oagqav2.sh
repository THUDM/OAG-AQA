model_cfg=/workspace/zfj/ssj/ptms/models--BAAI--llm-embedder
ctx_file=/workspace/zfj/ssj/dataset/oagqa-topic-v2
out_file=out_files
filename=ssj
checkpoint=/workspace/zfj/ssj/ptms/dpr_ckpt/hf_bert_base.cp

topic=$1
topk=$2


python3 generate_dense_embeddings.py \
	--pretrained_model_cfg $model_cfg \
	--model_file $checkpoint \
	--shard_id 0 --num_shards 1 \
	--ctx_file $ctx_file/$topic-papers-10k.tsv \
	--out_file encoded_files/encoded-oagqa-$topic-$filename \
	--batch_size 128 \
	--sequence_length 256  

python3 dense_retriever.py \
	--pretrained_model_cfg $model_cfg \
	--model_file $checkpoint \
	--ctx_file $ctx_file/$topic-papers-10k.tsv \
	--qa_file $ctx_file/$topic-questions.tsv \
	--encoded_ctx_file "encoded_files/encoded-oagqa-$topic-$filename-*.pkl" \
	--n-docs $topk \
	--batch_size 128 \
	--sequence_length 256 \
	--oagqa \
    --out_file $out_files/oagqa-$topic-result.json
