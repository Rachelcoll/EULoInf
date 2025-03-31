#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

dir_data="../data/"
dir_save_ckpt="../checkpoint/"
dir_save_tensors="../influences/"
dir_save_subset_indices="../data/subset_indices/"

data_name="sst2"
model_name="textcnn"
train_size=2000
batch_size=64

python3 ../src/train_model_get_scores.py \
        --data_name "$data_name" \
        --model_name "$model_name" \
        --dir_data "$dir_data" \
        --train_size "$train_size" \
        --batch_size "$batch_size" \
        --dir_save_subset_indices "$dir_save_subset_indices" \
        --num_epochs 100 \
        --ckpt_list 3 6 10 13 16 20 25 40 60 80 100 \
        --dir_save_ckpt "$dir_save_ckpt" \
        --dir_save_tensors "$dir_save_tensors" \
        --scoring_methods if

wait
echo "All done"