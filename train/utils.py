
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from torch_influence.base import BaseObjective
from torch_influence.modules import CGInfluenceModule, LiSSAInfluenceModule, AutogradInfluenceModule
from torch.utils.data import DataLoader

def save_checkpoint(model, optimizer, epoch, sum_epochs, filepath="../checkpoint"):
    completion_rate = epoch / sum_epochs
    checkpoint = {
        'epoch': epoch,
        'degree of completion': completion_rate,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    filename = f"{filepath}/{type(model).__name__}_epoch_{epoch}_{completion_rate}.pth"
    torch.save(checkpoint, filename)
    print(f"Checkpoint at {completion_rate} saved: {filename}")
    return filename
    
def load_checkpoint(model, optimizer, filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    completion_rate = checkpoint['degree of completion']
    print(f"Checkpoint loaded at epoch {epoch} with completion rate {completion_rate}")
    return model, optimizer, epoch, completion_rate


def train_func_classification(model, criterion, optimizer, train_loader, validation_loader, epochs):
    train_losses = []
    accuracies = []
    checkpoints = []
    
    for epoch in range(epochs):
        model.train()
        loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            loss += loss.item()
            
        loss = loss / len(train_loader)
        train_losses.append(loss)
            
        model.eval()
        correct_count = 0
        for x_val_batch, y_val_batch in validation_loader:
            y_pred = model(x_val_batch)
            predictions = torch.argmax(y_pred, dim=-1)
            result = predictions == y_val_batch
            correct_count += result.sum()
            
        accuracy = (correct_count / len(validation_loader.dataset)).item()
        accuracies.append(accuracy)
        print(f"Epoch: {epoch + 1}, Train Loss: {loss}, accuracy: {accuracy}")
        if (epoch + 1) / epochs in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            checkpoint_f = save_checkpoint(model, optimizer, epoch + 1, epochs, f"../checkpoint/{type(model).__name__}")
            checkpoints.append(checkpoint_f)
        
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 5))
    train_losses = [loss.detach().numpy() for loss in train_losses]
    max_accuracy = np.max(accuracies)
    min_accuracy = np.min(accuracies)
    accuracy_diff = max_accuracy - min_accuracy
    val_accuracies = [((accuracy - min_accuracy) / accuracy_diff) * (np.max(train_losses) - np.min(train_losses)) + np.min(train_losses) 
                      for accuracy in accuracies]
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.title('Train and Accuracy')
    plt.legend()
    plt.show()
    return model, checkpoints, train_losses, accuracies

def calculate_influence_score_nn(
    traindata: DataLoader, 
    testdata: DataLoader, 
    model, 
    criterion,
    device,
    epochs=-1,
    train_func=None,
    per_point=-1, 
    appro="cg"
    ):
    """
    Calculate influence score for each training data point
    Parameters
    ----------
    traindata: train data loader
    testdata: test data loader
    model: nn.Module
    per_point: if >= 0: return instance score for specific test point; else return instance score for test sets
    Returns
    -------

    """

    # initialize the model and optimizer
    trainidx = len(traindata.dataset)
    testidx = len(testdata.dataset)

    class NNObjective(BaseObjective):
        
        # batch: Dataset
        def train_outputs(self, model, batch):
            inputs, _ = batch
            inputs = inputs
            inputs = inputs.to(device, dtype=torch.float)
            return model(inputs)

        def train_loss_on_outputs(self, outputs, batch):
            _, labels = batch
            labels = labels
            labels = labels.to(device)
            return criterion(outputs, labels)

        def train_regularization(self, params):
            return 0

        # training loss by default taken to be
        # train_loss_on_outputs + train_regularization

        def test_loss(self, model, params, batch):
            inputs, labels = batch
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)
            return criterion(model(inputs), labels)
        
    if train_func != None:
        model, _, val_losses = train_func(
            model = model,
            criterion = criterion,
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001),
            train_loader = traindata,
            validation_loader = testdata,
            epochs = epochs,
            )
    
    print("Start calculating influence scores...")

    if appro == "lissa":
        print("LissaInfluenceModule")
        module = LiSSAInfluenceModule(
            model=model,
            objective=NNObjective(),
            train_loader=traindata,
            test_loader=testdata,
            device=device,
            damp=0.003,
            repeat=1,
            depth=2500,
            scale=1e4,
        )
    elif appro == "cg":
        print("CGInfluenceModule")
        module = CGInfluenceModule(
            model=model,
            objective=NNObjective(),
            train_loader=traindata,
            test_loader=testdata,
            device=device,
            damp=0.003,
            atol=1e-8,
            maxiter=1000,
        )
    elif appro == "autograd":
        print("AutogradInfluenceModule")
        module = AutogradInfluenceModule(
            model=model,
            objective=NNObjective(),
            train_loader=traindata,
            test_loader=testdata,
            device=device,
            damp=0.003
        )

    print(f"train data size: {len(traindata.dataset)}")
    print(f"test data size: {len(testdata.dataset)}")
    if train_func != None:
        print(f"model performance validation loss / accuracy: {val_losses[-1]}")
    
    if per_point >= 0:
        scores = module.influences(np.arange(trainidx),  np.array([per_point])).cpu().numpy().reshape(-1)
    else:
        scores = module.influences(np.arange(trainidx), np.arange(testidx)).cpu().numpy().reshape(-1)
    return scores

def influence_at_checkpoints(traindata, testdata, model, optimizer,
                             criterion, checkpoints, device, per_point=-1, appro="cg"):
    scores = []
    for checkpoint in tqdm(checkpoints, desc=f"Calculating influence scores"):
        model, optimizer, _, completion_rate = load_checkpoint(model, optimizer, checkpoint)
        print(f"completion_rate: {completion_rate}" )
        score = calculate_influence_score_nn(traindata, testdata, model, criterion, device, per_point=per_point, appro=appro)
        scores.append(score)
    return scores