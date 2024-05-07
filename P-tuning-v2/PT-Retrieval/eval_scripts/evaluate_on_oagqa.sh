model_cfg=/workspace/zfj/ssj/ptms/llm-embedder
filename=ptv2-dpr-multidata-128-40-8e-3-64
checkpoint=dpr_biencoder.39.1045
psl=64

topic=$1
topk=$2

python3 generate_dense_embeddings.py \
	--pretrained_model_cfg /workspace/zfj/ssj/ptms/llm-embedder \
	--model_file /workspace/zfj/ssj/ptms/llm-embedder \
	--shard_id 0 --num_shards 1 \
	--ctx_file /workspace/zfj/ssj/dataset/oagqa-topic-v2/$topic-papers-10k.tsv \
	--out_file encoded_files/encoded-oagqa-$topic-$filename \
	--batch_size 128 \
	--sequence_length 256  

python3 dense_retriever.py \
	--pretrained_model_cfg /workspace/zfj/ssj/ptms/llm-embedder \
	--model_file checkpoints/$filename/$checkpoint \
	--ctx_file /workspace/zfj/ssj/dataset/oagqa-topic-v2/$topic-papers-10k.tsv \
	--qa_file /workspace/zfj/ssj/dataset/oagqa-topic-v2/$topic-questions.tsv \
	--encoded_ctx_file "encoded_files/encoded-oagqa-$topic-$filename-*.pkl" \
	--n-docs $topk \
	--batch_size 128 \
	--sequence_length 256 \
	--oagqa  
