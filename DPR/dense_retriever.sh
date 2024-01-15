
CUDA_VISIBLE_DEVICES=2 python dense_retriever.py \
	model_file=/home/shishijie/workspace/project/DPR/checkpoints/stackex_qa_hn/dpr_biencoder.33 \
	qa_dataset=stackex_qa_test \
	ctx_datatsets=[dpr_stackex_qa] \
	encoded_ctx_files=[/home/shishijie/workspace/project/DPR/checkpoints/stackex_qa_hn/ctx_biencoder.33/ctx_biencoder.33_0] \
	out_file=/home/shishijie/workspace/project/DPR/checkpoints/stackex_qa_hn/ctx_biencoder.33/res.json