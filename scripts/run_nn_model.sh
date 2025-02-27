#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

dir_data="/home/nicole/Workspace/data_valuation/data/"
dir_save_ckpt="/home/nicole/Workspace/data_valuation/checkpoint/"
dir_save_tensors="/home/nicole/Workspace/data_valuation/influences/"

python3 ../src/train_nn_model.py \
        --dir_data "$dir_data" \
        --num_epochs 10 \
        --ckpt_min_max 3 10 \
        --ckpt_step 3 \
        --dir_save_ckpt "$dir_save_ckpt" \
        --dir_save_tensors "$dir_save_tensors"

wait
echo "All done"