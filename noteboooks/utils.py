import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from sklearn.decomposition import PCA

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


def create_influences_dict(save_objs_dict, ckpt_labels):
    validate_dict = {}
    rand_dict = {}
    for scoring_method in ["if", "tracin"]:
        validate_list = save_objs_dict[f"validate_list_{scoring_method}"]
        rand_list = save_objs_dict[f"rand_list_{scoring_method}"]

        validate_dict[f"validate_accuracies_{scoring_method}"] = []
        validate_dict[f"validate_train_losses_{scoring_method}"] = []
        validate_dict[f"validate_validation_losses_{scoring_method}"] = []

        rand_dict[f"rand_validate_accuracies_{scoring_method}"] = []
        rand_dict[f"rand_validate_train_losses_{scoring_method}"] = []
        rand_dict[f"rand_validate_validation_losses_{scoring_method}"] = []

        for ckpt_idx, _ in enumerate(ckpt_labels):
            validate_dict[f"validate_accuracies_{scoring_method}"].append(validate_list[ckpt_idx].accuracies)
            validate_dict[f"validate_train_losses_{scoring_method}"].append(validate_list[ckpt_idx].train_losses)
            validate_dict[f"validate_validation_losses_{scoring_method}"].append(validate_list[ckpt_idx].validation_losses)

            rand_dict[f"rand_validate_accuracies_{scoring_method}"].append(rand_list[ckpt_idx].accuracies)
            rand_dict[f"rand_validate_train_losses_{scoring_method}"].append(rand_list[ckpt_idx].train_losses)
            rand_dict[f"rand_validate_validation_losses_{scoring_method}"].append(rand_list[ckpt_idx].validation_losses)
    return validate_dict, rand_dict

def top_last_influences(ckpt_influences, train_embeds, num_per_row=10, descending=True):
    top_inf_data = []
    for influences in ckpt_influences:
        sorted_influences = torch.argsort(torch.from_numpy(influences), descending=descending)
        top_indexes = sorted_influences[:num_per_row]
        top_influential_train_data = [(train_embeds[index][0].detach().squeeze(0), train_embeds[index][1]) for index in top_indexes]
        top_inf_data.append(top_influential_train_data)
    return top_inf_data

def random_idx(checkpoint_influences, dataset, num_data, p):
    checkpoints_indices = []
    for _ in checkpoint_influences:
        num_select = int(p * num_data)
        indices = list(np.arange(len(dataset)))
        random_indices = random.sample(indices, num_select)
        checkpoints_indices.append(random_indices)
    return checkpoints_indices

def concat_ckpt_metrics(metrics, ckpt_labels):
    integrate_metrics = []
    for i, metric in enumerate(metrics):
        prev_ckpt = ckpt_labels[i-1] if i > 0 else 0
        integrate_metrics.append(metric[prev_ckpt:])
    return np.concatenate(integrate_metrics)

def calc_inter_dist(embeds):
    distances = euclidean_distances(embeds, embeds)
    mean_distance = np.mean(distances)
    return mean_distance

def calc_external_dist(inf_embeds, other_embeds):
    distances = euclidean_distances(inf_embeds, other_embeds)
    mean_distance = np.mean(distances)
    return mean_distance

def calc_pairwise_euclideans(influences, train_subset, list_p, transform=None):
    num_data = len(train_subset)
    data_list = []
    for p in list_p:
        int_distances = []
        top_inf_embeds = top_last_influences(influences, train_subset, int(p*num_data))
        for ckpt, ckpt_embeds in enumerate(top_inf_embeds):
            embed_inputs = np.array([embed[0] for embed in ckpt_embeds])
            embed_inputs = transform(embed_inputs) if transform else embed_inputs
            mean_dist = calc_inter_dist(embed_inputs)
            int_distances.append(mean_dist)
        data_list.append(int_distances)
    return data_list

def random_p_data(checkpoint_influences, train_dataset, num_data, p):
    checkpoints_data = []
    for _ in checkpoint_influences:
        num_select = int(p * num_data)
        random_data = random.sample(list(train_dataset), num_select)
        random_data = [(data[0].detach().numpy().squeeze(0), data[1]) for data in random_data]
        checkpoints_data.append(random_data)
    return checkpoints_data

def create_inf_int_distances(ckpt_embds):
    inf_int_distances = []
    for i in range(len(ckpt_embds)):
        embeds = np.array([embed[0] for embed in ckpt_embds[i]])[:, 0, :]
        repre = calc_inter_dist(embeds)
        inf_int_distances.append(repre)
    return inf_int_distances

def calc_dist_inf_to_others(select_inf_embeds, other_inf_embeds):
    distances = []
    for i, ckpt_embeds in enumerate(select_inf_embeds):
        inf_embeds =  np.array([embed[0] for embed in ckpt_embeds])[:, 0, :]
        rand_embeds =  np.array([embed[0] for embed in other_inf_embeds[i]])[:, 0, :]
        mean_dist = calc_external_dist(inf_embeds, rand_embeds)
        distances.append(mean_dist)
    return distances


def inter_sentence_entropy(sentence_embeds):
    pairwise_distances = pdist(sentence_embeds, metric='cosine')
    distance_probs, _ = np.histogram(pairwise_distances, bins=30, density=True)
    distance_probs = distance_probs / np.sum(distance_probs)
    pairwise_entropy = entropy(distance_probs)
    return pairwise_entropy

def create_inter_entropies(ls_ckpt_embds):
    entropies = []
    for ckpt, ckpt_embeds in enumerate(ls_ckpt_embds):
        embeds = np.array([embed[0] for embed in ckpt_embeds])[:, 0, :]
        batch_entropy = inter_sentence_entropy(embeds)
        entropies.append(batch_entropy)
    return entropies

def inner_entropy(sentence_embed):
    mean_embed = sentence_embed[0, :]
    probabilities = torch.nn.functional.softmax(torch.tensor(mean_embed), dim=0).numpy()
    inner_entropy = entropy(probabilities)
    return inner_entropy

def create_inner_entropies(ls_ckpt_embds):
    entropies = []
    for ckpt, ckpt_embeds in enumerate(ls_ckpt_embds):
        embed_entropies = [inner_entropy(embed[0]) for embed in ckpt_embeds]
        entropies.append(np.mean(embed_entropies))
    return entropies

def volume(sentences_embeds):
    return np.sqrt(np.linalg.det(sentences_embeds.T@sentences_embeds))

def pca(x, k):
    x_std = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)
    pca = PCA(n_components=k)
    pca.fit(x_std)
    return pca.components_

def calc_volumes(ls_ckpts_embeds):
    data_list = []
    for i, embeds in enumerate(ls_ckpts_embeds):
        volumes = []
        for ckpt_embeds in embeds:
            mean_embeds = np.array([embed[0] for embed in ckpt_embeds])[:, 0, :]
            W = pca(mean_embeds, 10)
            Z_embeds = mean_embeds @ W.T
            volumes.append(volume(Z_embeds))
        data_list.append(volumes)
    return data_list


