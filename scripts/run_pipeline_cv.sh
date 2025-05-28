#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

dir_data="../data/"
dir_save_ckpt="../checkpoint/"
dir_save_tensors="../influences/"
dir_save_subset_indices="../data/subset_indices/"

data_name="dogfish"
model_name="v3fc"

/d/anaconda3/envs/data-valuation-env/python ../src/train_model_get_scores.py \
        --data_name "$data_name" \
        --model_name "$model_name" \
        --dir_data "$dir_data" \
        --dir_save_subset_indices "$dir_save_subset_indices" \
        --num_epochs 100 \
        --ckpt_list 1 2 3 5 10 15 20 40 60 80 100 \
        --dir_save_ckpt "$dir_save_ckpt" \
        --dir_save_tensors "$dir_save_tensors" \
        --scoring_methods  tracin \
        --appro no_hessian \

wait
echo "All done"