CUDA_VISIBLE_DEVICES=2 python train_dense_encoder.py \
    train_datasets=[stackex_qa_train] \
    dev_datasets=[stackex_qa_valid]
    train=biencoder_local \
    output_dir=/workspace/project/DPR/checkpoints/stackex_qa_hn/