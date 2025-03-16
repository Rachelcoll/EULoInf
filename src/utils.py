
import os
import torch
import random
import numpy as np
from tqdm.auto import tqdm
from logging_config import logger
from torch.utils.data import DataLoader, Subset
from response_types import ModelEvalObjects

def random_idx(checkpoint_influences, dataset, num_data, p):
    checkpoints_indices = []
    for _ in checkpoint_influences:
        num_select = int(p * num_data)
        indices = list(np.arange(len(dataset)))
        random_indices = random.sample(indices, num_select)
        checkpoints_indices.append(random_indices)
    return checkpoints_indices

def save_checkpoint(model, optimizer, epoch, sum_epochs, dir_ckpt):
    completion_rate = epoch / sum_epochs
    checkpoint = {
        'epoch': epoch,
        'degree of completion': completion_rate,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    filename = os.path.join(dir_ckpt, f"{type(model).__name__}_epoch_{epoch}_{completion_rate}.pth")
    torch.save(checkpoint, filename)
    print(f"Checkpoint at {completion_rate} saved: {filename}")
    return filename
    
def load_checkpoint(model, optimizer, filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    completion_rate = checkpoint['degree of completion']
    # print(f"Checkpoint loaded at epoch {epoch} with completion rate {completion_rate}")
    return model, optimizer, epoch, completion_rate


def train_func_classification(model, criterion, optimizer, train_loader, validation_loader, epochs, predict_func, device, record_checkpoints=None, dir_ckpt="", save=True):
    train_losses = []
    accuracies = []
    checkpoint_filepaths = []
    validation_losses = []
    
    for epoch in range(epochs):
        model.train()
        loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            loss += loss.item()
            
        loss = loss / len(train_loader)
        train_losses.append(loss)
            
        model.eval()
        val_loss = 0.0
        correct_count = 0
        for x_val_batch, y_val_batch in validation_loader:
            x_val_batch = x_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            y_pred = model(x_val_batch)
            predictions = predict_func(y_pred)
            result = predictions == y_val_batch
            correct_count += result.sum()
            val_loss += criterion(y_pred, y_val_batch).item()
        val_loss_mean = val_loss / len(validation_loader)
        validation_losses.append(val_loss_mean)
            
        accuracy = (correct_count / len(validation_loader.dataset)).item()
        accuracies.append(accuracy)
        if save:
            logger.debug(f"Epoch: {epoch + 1}, Train Loss: {loss}, accuracy: {accuracy}")
            if record_checkpoints != None:
                if (epoch + 1) in record_checkpoints:
                    checkpoint_f = save_checkpoint(model, optimizer, epoch + 1, epochs, dir_ckpt)
                    checkpoint_filepaths.append(checkpoint_f)
            else:
                if (epoch + 1) / epochs in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    checkpoint_f = save_checkpoint(model, optimizer, epoch + 1, epochs, dir_ckpt)
                    checkpoint_filepaths.append(checkpoint_f)
    train_losses = [loss.item() for loss in train_losses]
    # return model, checkpoint_filepaths, train_losses, accuracies, validation_losses
    return checkpoint_filepaths, ModelEvalObjects(train_losses = train_losses, accuracies = accuracies, validation_losses = validation_losses)



def random_p_data(checkpoint_influences, train_dataset, num_data, p=0.01):
    checkpoints_data = []
    for _ in tqdm(checkpoint_influences):
        num_select = int(p * num_data)
        random_data = random.sample(list(train_dataset), num_select)
        checkpoints_data.append(random_data)
    return checkpoints_data


def rest_data(ckpt_influences, inf_num, train_dataset):
    ckpt_imgs = []
    for _, influences in enumerate(ckpt_influences):
        sorted_influences = torch.argsort(torch.from_numpy(influences), descending=True)
        rest_indexes = sorted_influences[inf_num:]
        top_influential_train_data = [train_dataset[index] for index in rest_indexes]
        imgs = []
        for i, (img, label) in enumerate(top_influential_train_data):
            imgs.append(np.array(img))        
        ckpt_imgs.append(imgs)
    return ckpt_imgs


def validate_influence(influences, train_dataset, test_dataloader, model_class, criterion, device, checkpoint_filepaths, checkpoint_labels, p, predict_func):
    # top_ckpts_embeds = top_last_influences(influences, train_dataset, int(p*num_data))
    num_data = len(train_dataset)
    random_indices = random_idx(influences, train_dataset, num_data, 1 - p)
    validate_list = []
    rand_list = []
    
    for i in tqdm(range(len(checkpoint_filepaths)), desc=f"Validating influences"):
        sorted_influences = torch.argsort(torch.from_numpy(influences[i]), descending=True)
        top_indexes = sorted_influences[:int(p*num_data)]
        filtered_indices = [i for i in list(range(len(train_dataset))) if i not in top_indexes]
        residual_train_dataset = Subset(train_dataset, filtered_indices)
        residual_dataloader = DataLoader(residual_train_dataset, batch_size=20, shuffle=True)
        
        random_dataset = Subset(train_dataset, random_indices[i])
        random_dataloader = DataLoader(random_dataset, batch_size=64, shuffle=True)
        
        epochs = checkpoint_labels[i]

        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        _, val_model_eval_obj = train_func_classification(
            model, criterion, optimizer, residual_dataloader, test_dataloader, epochs, predict_func, device, save=False)
        validate_list.append(val_model_eval_obj)
        
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        _, rand_model_eval_obj = train_func_classification(
            model, criterion, optimizer, random_dataloader, test_dataloader, epochs, predict_func, device, save=False)
        rand_list.append(rand_model_eval_obj)
        
    return validate_list, rand_list
    # return validate_train_losses, validate_accuracies, validate_validation_losses, rand_validate_train_losses, rand_validate_accuracies, rand_validate_validation_losses
