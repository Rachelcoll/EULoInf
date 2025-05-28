import torch
import numpy as np
from tqdm.auto import tqdm
from logging_config import logger
from utils import load_checkpoint
from torch.utils.data import DataLoader, Dataset

def influence_at_checkpoints_tracin(traindata, testdata, model, criterion, checkpoint_filepaths, lr=0.001):
    scores = []
    ckpt_filepaths = []
    for i, checkpoint in enumerate(tqdm(checkpoint_filepaths, desc=f"Calculating influence scores with Tracin")):
        ckpt_filepaths.append(checkpoint)
        score = calculate_tracin_score(model, ckpt_filepaths, traindata, testdata, criterion, lr)
        scores.append(score.cpu().numpy())
    return scores

def influence_at_checkpoints_tracin_updated(traindata, testdata, model, criterion, checkpoint_filepaths, lr=0.001):
    scores = []
    for i, checkpoint in enumerate(tqdm(checkpoint_filepaths, desc=f"Calculating influence scores with Tracin")):
        score = calculate_tracin_score_test_first(model, checkpoint, traindata, testdata, criterion, lr)
        scores.append(score.cpu().numpy())
    return scores

def flatten_params_like(params_like):
    vector = []
    for p in params_like:
        vector.append(p.view(-1))
    return torch.cat(vector)

def create_batch_grads(model, checkpoint, criterion, x_batch, y_batch):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model, _, _, _ = load_checkpoint(model, optimizer, checkpoint)
    batch_size = len(x_batch)
    y_pred = model(x_batch)
    loss = criterion(y_pred, y_batch)
    # loss.backward()
    # grad_sample = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1)
    #                          for param in model.parameters()])
    params = tuple(p for name, p in model.named_parameters() if p.requires_grad)
    params_grad = torch.autograd.grad(loss, params)
    grad_batch = flatten_params_like(params_grad)

    return grad_batch, batch_size


def calculate_tracin_score(model, ckpt_filepaths, train_dataloader, test_dataloader, criterion, lr):
    device = model.device
    lr = torch.tensor(lr, device=device)
    scores_matrix_list = []
    
    for train_batch_id, (x_train, y_train) in enumerate(tqdm(train_dataloader, desc=f"Getting grads for train batch...")):
        for test_batch_id, (x_test, y_test) in enumerate(test_dataloader):
            grad_sum = torch.zeros(x_train.shape[0], x_test.shape[0], device=device)
            for ckpt_file in ckpt_filepaths:
                w = torch.load(ckpt_file, map_location=device)["model_state_dict"]

                train_grad_batch =  create_batch_grads(model, w, criterion, x_train, y_train) # gets out (num_in_batch, param size)
                test_grad_batch =  create_batch_grads(model, w, criterion, x_test, y_test) # gets out (num_in_batch, param size)
                grad_sum +=  (lr * train_grad_batch @ test_grad_batch.T)
            
            scores_matrix_list.append(grad_sum)
    
    # Format the final matrix to be (num_data_train, num_data_test)
    tracins_scores = format_matrices(len(train_dataloader), len(test_dataloader), scores_matrix_list)
    
    # Avg influence scores of a set of each train data point wrt (mean) test loss over set of test data points
    tracins_scores = tracins_scores.mean(dim=1) 
    # tracins_scores = tracins_scores.max(dim=1).values
    return tracins_scores

def calculate_tracin_score_test_first(model, checkpoint, train_dataloader, test_dataloader, criterion, lr):
    device = model.device
    lr = torch.tensor(lr, device=device)
    train_dp_list = []
    dataset = train_dataloader.dataset
    new_train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        collate_fn=train_dataloader.collate_fn,
        num_workers=train_dataloader.num_workers,
        worker_init_fn=train_dataloader.worker_init_fn,
    )

    # w = torch.load(checkpoint, map_location=device)["model_state_dict"]
    test_grad = 0.0
    for x_test, y_test in tqdm(test_dataloader, desc=f"Getting grads for test batch..."):
        test_grad_batch, batch_size = create_batch_grads(model, checkpoint, criterion, x_test, y_test)
        test_grad += test_grad_batch * batch_size
    test_grads_mean = test_grad / len(test_dataloader.dataset)

    for x_train, y_train in tqdm(new_train_loader, desc=f"Getting grads for train batch..."):
        train_grad_batch, batch_size = create_batch_grads(model, checkpoint, criterion, x_train, y_train)
        dot_product = (train_grad_batch @ test_grads_mean) * lr
        train_dp_list.append(dot_product)

    tracins_scores = torch.tensor(train_dp_list) / len(train_dataloader.dataset)
    return tracins_scores



def format_matrices(num_train_batches, num_test_batches, scores_matrix_list):
    final_matrix_list = []

    for train_batch_idx in range(num_train_batches):
        # Split the list into `x` chunks, each containing `k` items
        matrices = scores_matrix_list[train_batch_idx * num_test_batches: (train_batch_idx + 1) * num_test_batches]
        matrices = torch.cat(matrices, dim=1)
        final_matrix_list.append(matrices)

    tracin_scores = torch.cat(final_matrix_list, dim=0)
    return tracin_scores 