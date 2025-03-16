import torch
import numpy as np
from tqdm.auto import tqdm
from logging_config import logger

def influence_at_checkpoints_tracin(traindata, testdata, model, criterion, checkpoint_filepaths, lr=0.001):
    scores = []
    ckpt_filepaths = []
    for i, checkpoint in enumerate(tqdm(checkpoint_filepaths, desc=f"Calculating influence scores with Tracin")):
        if len(ckpt_filepaths) == 0:
            ckpt_filepaths.append(checkpoint)
        else:
            ckpt_filepaths = [ckpt_filepaths[-1]] + [checkpoint]
        score = calculate_tracin_score(model, ckpt_filepaths, traindata, testdata, criterion, lr)
        scores.append(score.cpu().numpy())
    return scores

def create_batch_grads(model, w, criterion, x_batch, y_batch):
    device = model.device
    batch_size = len(x_batch)
    grad_batch = []
    for idx in range(batch_size):
        model.load_state_dict(w)
        model.eval()

        input = x_batch[idx].unsqueeze(dim=0).to(device) # make it batch_size = 1
        gt = y_batch[idx].unsqueeze(dim=0).to(device) # make it batch_size = 1

        y_pred = model(input)
        loss = criterion(y_pred, gt)
        loss.backward()
        grad_sample = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
        grad_batch.append(grad_sample)
    grad_batch = torch.stack(grad_batch, dim=0)
    return grad_batch


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
                test_grad_batch =  create_batch_grads(model, w, criterion, x_test, y_test)# gets out (num_in_batch, param size)
                grad_sum +=  (lr * train_grad_batch @ test_grad_batch.T)
            
            scores_matrix_list.append(grad_sum)
    
    # Format the final matrix to be (num_data_train, num_data_test)
    tracins_scores = format_matrices(len(train_dataloader), len(test_dataloader), scores_matrix_list)
    
    # Avg influence scores of a set of each train data point wrt (mean) test loss over set of test data points
    tracins_scores = tracins_scores.mean(dim=1) 
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