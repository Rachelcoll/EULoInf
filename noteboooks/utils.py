import numpy as np
import torch
import random
from tqdm import tqdm

def pairwise_similarity(ckpts_indices):
    similarities = []
    for i in range(len(ckpts_indices)):
        similarity = []
        for j in range(len(ckpts_indices)):
            common = len(set(ckpts_indices[i]).intersection(set(ckpts_indices[j])))
            union = len(set(ckpts_indices[i]).union(set(ckpts_indices[j])))
            similarity.append(common / union)
        similarities.append(similarity)
    return np.array([np.array(sim) for sim in similarities])

def top_inf_indices(ckpts_influences, num_per_row):
    top_inf_indices = []
    for ckpt_influences in ckpts_influences:
        sorted_influences = torch.argsort(torch.from_numpy(ckpt_influences), descending=True)
        top_indexes = sorted_influences[:num_per_row].tolist()
        top_inf_indices.append(top_indexes)
    return top_inf_indices

def random_p_data(checkpoint_influences, train_dataset, num_data, p=0.01):
    checkpoints_data = []
    for _ in tqdm(checkpoint_influences):
        num_select = int(p * num_data)
        random_data = random.sample(list(train_dataset), num_select)
        checkpoints_data.append(random_data)
    return checkpoints_data