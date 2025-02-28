#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

dir_data="../data/"
dir_save_ckpt="../checkpoint/"
dir_save_tensors="../influences/"
dir_save_subset_indices="../data/subset_indices/"

python3 ../src/train_nn_model.py \
        --dir_data "$dir_data" \
        --num_epochs 10 \
        --ckpt_list 3 10 \
        --dir_save_ckpt "$dir_save_ckpt" \
        --dir_save_tensors "$dir_save_tensors" \
        --dir_save_subset_indices "$dir_save_subset_indices" \
        --sample_size 8000

wait
echo "All done"