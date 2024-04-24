model_cfg=bert-base-uncased
psl=64
checkpoint=ptv2.colbert.msmarco.cosine.1e-3.64
filename=$checkpoint/colbert-360000.dnn
dir=oagqa_eval_lists/*

for f in $dir
do
    echo $f
    echo ${f#*/}
    python3 batch_dense_retrieval_colbert.py \
        --pretrained_model_cfg $model_cfg \
        --checkpoint checkpoints/$filename \
        --prefix --pre_seq_len $psl \
        --ctx_dir data/oagqa-topic-v2 \
        --out_dir encoded_colbert_files \
        --input_file $f \
        --result_file "results/ptv2-colbert/oagqa_results_${f#*/}.json" \
        --amp --doc_maxlen 300  --mask-punctuation --bsize 256 \
        --batch_size 100 \
        --sequence_length 256 \
        --bs 4 --oagqa \
        --n-docs 20
done
