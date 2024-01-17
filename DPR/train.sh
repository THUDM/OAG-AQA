
CUDA_VISIBLE_DEVICES=2 python train_dense_encoder.py \
    train_datasets=[stackex_qa_train] \
    dev_datasets=[stackex_qa_valid] \
    train=biencoder_local \
    output_dir=/home/shishijie/workspace/project/DPR/checkpoints/stackex_qa_hn/

# CUDA_VISIBLE_DEVICES=0,1,2,4 python -m torch.distributed.launch --nproc_per_node=4 \
#     train_datasets=[stackex_qa_train] \
#     train=biencoder_local \
#     output_dir=/home/shishijie/workspace/project/DPR/checkpoints/stackex_qa_hn_multi/

