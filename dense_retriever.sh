model_dir=/home/zhangfanjin/projects/qa/OAG-AQA/outputs/2024-04-15/15-27-08/output_dpr
epoch=29
python dense_retriever.py \
	model_file=$model_dir/dpr_biencoder.$epoch \
	qa_dataset=stackex_qa_test \
	ctx_datatsets=[dpr_stackex_qa] \
	encoded_ctx_files=[$model_dir/ctx_encoder_$epoch.pkl_0] \
	out_file=$model_dir/valid_sub_ckpt_$epoch.txt
