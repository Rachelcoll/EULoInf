import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.utils import load_checkpoint
from scipy.interpolate import griddata
import matplotlib.patches as mpatches

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

def top_last_influences(ckpt_influences, train_dataset, num_per_row=10, descending=True):
    ckpts_top_inf_data = []
    for influences in ckpt_influences:
        sorted_influences = torch.argsort(torch.from_numpy(influences), descending=descending)
        top_indexes = sorted_influences[:num_per_row]
        top_influential_train_data = [(train_dataset[index][0].squeeze(), train_dataset[index][1]) for index in top_indexes]
        ckpts_top_inf_data.append(top_influential_train_data)
    return ckpts_top_inf_data

def top_inf_indices(ckpts_influences, num_per_row):
    top_inf_indices = []
    for ckpt_influences in ckpts_influences:
        sorted_influences = torch.argsort(torch.from_numpy(ckpt_influences), descending=True)
        top_indexes = sorted_influences[:num_per_row].tolist()
        top_inf_indices.append(top_indexes)
    return top_inf_indices

# def random_p_data(checkpoint_influences, train_dataset, num_data, p=0.01):
#     checkpoints_data = []
#     for _ in tqdm(checkpoint_influences):
#         num_select = int(p * num_data)
#         random_data = random.sample(list(train_dataset), num_select)
#         checkpoints_data.append(random_data)
#     return checkpoints_data

def random_p_data(checkpoint_influences, train_dataset, num_data, p):
    checkpoints_data = []
    for _ in checkpoint_influences:
        num_select = int(p * num_data)
        random_data = random.sample(list(train_dataset), num_select)
        random_data = [(data[0].detach().numpy().squeeze(0), data[1]) for data in random_data]
        checkpoints_data.append(random_data)
    return checkpoints_data

def create_influences_dict(save_objs_dict, ckpt_labels, scoring_methods):
    validate_dict = {}
    rand_dict = {}
    for scoring_method in scoring_methods:
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

def calc_pairwise_euclideans(influences, train_subset, list_p, transform=None, option="nn"):
    num_data = len(train_subset)
    data_list = []
    for p in list_p:
        int_distances = []
        top_inf_embeds = top_last_influences(influences, train_subset, int(p*num_data))
        for ckpt, ckpt_embeds in enumerate(top_inf_embeds):
            embed_inputs = np.array([embed[0] for embed in ckpt_embeds])
            if (option == "nn") or (option == "cv"):
                 embed_inputs = transform(embed_inputs) if transform else embed_inputs
            elif option == "textcnn":
                embed_inputs = embed_inputs[:, 0, :]
            mean_dist = calc_inter_dist(embed_inputs)
            int_distances.append(mean_dist)
        data_list.append(int_distances)
    return data_list

def create_inf_int_distances(ckpt_embds, transform=None, option="nn"):
    inf_int_distances = []
    for i in range(len(ckpt_embds)):
        if (option == "nn") or (option == "cv"):
            embeds = np.array([embed[0] for embed in ckpt_embds[i]])
            embeds = transform(embeds) if transform else embeds
        elif option == "textcnn":
            embeds = np.array([embed[0] for embed in ckpt_embds[i]])[:, 0, :]
        repre = calc_inter_dist(embeds)
        inf_int_distances.append(repre)
    return inf_int_distances

def calc_dist_inf_to_others(select_inf_embeds, other_inf_embeds, transform=None, option="nn"):
    distances = []
    for i, ckpt_embeds in enumerate(select_inf_embeds):
        if (option == "nn") or (option == "cv"):
            inf_embeds =  np.array([embed[0] for embed in ckpt_embeds])
            inf_embeds = transform(inf_embeds) if transform else inf_embeds
            rand_embeds =  np.array([embed[0] for embed in other_inf_embeds[i]])
            rand_embeds = transform(rand_embeds) if transform else rand_embeds
        elif option=="textcnn":
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

def create_inter_entropies(ls_ckpt_embds, transform=None, option="nn"):
    entropies = []
    for ckpt, ckpt_embeds in enumerate(ls_ckpt_embds):
        if (option == "nn") or (option == "cv"):
            embeds = np.array([embed[0] for embed in ckpt_embeds])
            embeds = transform(embeds) if transform else embeds
        elif option=="textcnn":
            embeds = np.array([embed[0] for embed in ckpt_embeds])[:, 0, :]
        batch_entropy = inter_sentence_entropy(embeds)
        entropies.append(batch_entropy)
    return entropies

def inner_entropy(embed):
    probabilities = torch.nn.functional.softmax(torch.tensor(embed), dim=0).numpy()
    inner_entropy = entropy(probabilities)
    return inner_entropy

def create_inner_entropies(ls_ckpt_embds, transform=None):
    entropies = []
    for ckpt, ckpt_embeds in enumerate(ls_ckpt_embds):
        if transform:
            embed_entropies = [inner_entropy(transform(embed[0])) for embed in ckpt_embeds]
        else:
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

def calc_volumes(ls_ckpts_embeds, dim=10, transform=None, option="nn"):
    data_list = []
    for i, embeds in enumerate(ls_ckpts_embeds):
        volumes = []
        for ckpt_embeds in embeds:
            if (option == "nn") or (option == "cv"):
                mean_embeds = transform(np.array([embed[0] for embed in ckpt_embeds])) if transform else np.array([embed[0] for embed in ckpt_embeds])
            elif option=="textcnn":
                mean_embeds = np.array([embed[0] for embed in ckpt_embeds])[:, 0, :]
            W = pca(mean_embeds, dim)
            Z_embeds = mean_embeds @ W.T
            volumes.append(volume(Z_embeds))
        data_list.append(volumes)
    return data_list

def tsne_all_dataset(train_dataset, transform=None):
    all_data_x = np.array([embed[0] for embed in train_dataset])
    all_data_y = np.array([embed[1] for embed in train_dataset], dtype=np.int32)
    tsne = TSNE(n_components=2, random_state=0, perplexity=min(1, len(all_data_x)))
    if transform:
        all_data_x = transform(all_data_x) if transform else all_data_x
    all_x_2d = tsne.fit_transform(all_data_x)
    return all_x_2d, all_data_y, all_data_x

def plt_decision_boundary(
    top_inf_indices,
    checkpoint_labels,
    all_x_2d, 
    all_data_y,
    all_y_pred,
    ax, i, 
    custom_cmap,
    norm
    ):
    top_inf_x_2d = all_x_2d[top_inf_indices]
    top_inf_y_real = all_data_y[top_inf_indices]
    sc = ax.scatter(top_inf_x_2d[:, 0], top_inf_x_2d[:, 1], 
                c=top_inf_y_real, cmap=custom_cmap, norm=norm, edgecolors='k')
    # plt.scatter(all_x_2d[:, 0], all_x_2d[:, 1], c=all_data_y, cmap='viridis', alpha=0.5)

    x_min, x_max = all_x_2d[:, 0].min() - 1, all_x_2d[:, 0].max() + 1
    y_min, y_max = all_x_2d[:, 1].min() - 1, all_x_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                        np.linspace(y_min, y_max, 300))
    grid_pred = griddata(all_x_2d, all_y_pred, (xx, yy), method='nearest')

    ax.contourf(xx, yy, grid_pred, alpha=0.3, cmap=custom_cmap, norm=norm)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    class_patches = [mpatches.Patch(color=custom_cmap(norm(label)), label=f"Class {label}") for label in range(custom_cmap.N)]
    legend_points = ax.legend(*sc.legend_elements(), loc="upper right", title=f"Influential Points")
    legend_background = ax.legend(handles=class_patches, loc="upper left", title=f"checkpoint {checkpoint_labels[i]} labels")
    ax.add_artist(legend_background)
    return ax
