import numpy as np
import torch
from utils import load_checkpoint
from torch.utils.data import DataLoader


def select_in_range_data(
        model,
        checkpoint,
        train_dataloader,
        percentile=5,
        activation_func="sigmoid",
        device=torch.device("cpu"),
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model, _, _, _ = load_checkpoint(model, optimizer, checkpoint)
    # output = 0.0
    distances = []
    for batch_x, batch_y in train_dataloader:
        batch_output = model(batch_x.to(device, dtype=torch.float))
        batch_output = batch_output.cpu().detach().numpy()
        if activation_func == "sigmoid":
            distances.extend(np.abs(batch_output - 0.5).flatten())
        elif activation_func == "softmax":
            sorted_output = np.sort(batch_output, axis=1)
            distances.extend((sorted_output[:, -1] - sorted_output[:, -2]).flatten())

    distances = np.argsort(distances)
    support_indices = distances[:int(len(distances) * percentile / 100)]
    return support_indices

def list_support_data(
        model,
        checkpoint_filepaths, 
        train_dataloader,
        lower_bound=20,
        upper_bound=50,
        activation_func="sigmoid",
        device=torch.device("cpu"),
):
    indices_dict_p = {}
    for p in range(lower_bound, upper_bound+1, 5):
        indices_list = []
        for i, filepath in enumerate(checkpoint_filepaths):
            support_indices = select_in_range_data(
                model, 
                filepath, 
                train_dataloader, 
                percentile=p,
                activation_func=activation_func, 
                device=device
            )

            indices_list.append(support_indices)
        indices_dict_p[p] = indices_list
    return indices_dict_p
        

# def select_support_data(
#         model,
#         checkpoint,
#         train_dataloader,
#         delta=0.1,
#         activation_func="sigmoid",
#         device=torch.device("cpu"),
# ):
#     support_indices = []
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#     model, _, _, _ = load_checkpoint(model, optimizer, checkpoint)
#     dataset = train_dataloader.dataset
#     new_train_loader = DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         sampler=None,
#         collate_fn=train_dataloader.collate_fn,
#         num_workers=train_dataloader.num_workers,
#         worker_init_fn=train_dataloader.worker_init_fn,
#     )

#     for index, (batch_x, _) in enumerate(new_train_loader):
#         output = model(batch_x.to(device, dtype=torch.float))
#         output = output.cpu().detach().numpy()
#         if activation_func == "sigmoid":
#             distance = np.abs(output - 0.5)
#             if distance < delta:
#                 support_indices.append(index)

#         elif activation_func == "softmax":
#             sorted_output = np.sort(output, axis=1)
#             distance = sorted_output[:, -1] - sorted_output[:, -2]
#             if distance < delta:
#                 support_indices.append(index)

#     return support_indices