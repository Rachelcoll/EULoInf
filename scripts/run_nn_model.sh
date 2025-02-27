#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1

dir_data="../data/"
dir_save_ckpt="../checkpoint/"
dir_save_tensors="../influences/"

/d/anaconda3/envs/CS2109S/python ../src/train_nn_model.py \
        --dir_data "$dir_data" \
        --num_epochs 100 \
        --ckpt_list 1 2 3 5 7 10 15 20 40 60 80 100 \
        --dir_save_ckpt "$dir_save_ckpt" \
        --dir_save_tensors "$dir_save_tensors" \
        --sample_size 10000

wait
echo "All done"