import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
from torch_influence.base import BaseObjective
from torch_influence.modules import CGInfluenceModule, LiSSAInfluenceModule, AutogradInfluenceModule
import matplotlib.pyplot as plt

def calculate_influence_score_nn(traindata: data.DataLoader, testdata: data.DataLoader, train_func, model, device, per_point=-1, appro="cg"):
    """
    Calculate influence score for each training data point
    Parameters
    ----------
    traindata: train data loader
    testdata: test data loader
    train_func: unit that can train the model
    model: nn.Module
    per_point: if >= 0: return instance score for specific test point; else return instance score for test sets
    Returns
    -------

    """

    # initialize the model and optimizer
    trainidx = len(traindata.dataset)
    testidx = len(testdata.dataset)

    class MyObjective(BaseObjective):
        
        # batch: Dataset
        def train_outputs(self, model, batch):
            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            mask_pos = batch['mask_pos'].to(device, dtype=torch.long)
            if 'llm' not in model._name_.lower():
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                return model(input_ids, attention_mask, token_type_ids, mask_pos)
            else:
                return model(input_ids, attention_mask, mask_pos)

        def train_loss_on_outputs(self, outputs, batch):
            label = batch['label'].to(device, dtype=torch.long)
            return nn.CrossEntropyLoss()(outputs, label)  # mean reduction required

        def train_regularization(self, params):
            return 0

        # training loss by default taken to be
        # train_loss_on_outputs + train_regularization

        def test_loss(self, model, params, batch):
            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            mask_pos = batch['mask_pos'].to(device, dtype=torch.long)
            label = batch['label'].to(device, dtype=torch.long)
            if 'llm' not in model._name_.lower():
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                return nn.CrossEntropyLoss()(model(input_ids, attention_mask, token_type_ids, mask_pos),
                                             label)  # no regularization in test loss
            else:
                return nn.CrossEntropyLoss()(model(input_ids, attention_mask, mask_pos), label)

    _, model = train_func.train_until_convergence(model, traindata, testdata,
                                                  gradient_steps_between_eval=2 * len(traindata))
    print("Start calculating influence scores...")

    if appro == "lissa":
        print("LissaInfluenceModule")
        module = LiSSAInfluenceModule(
            model=model,
            objective=MyObjective(),
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
            objective=MyObjective(),
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
            objective=MyObjective(),
            train_loader=traindata,
            test_loader=testdata,
            device=device,
            damp=0.003
        )

    print(f"train data size: {len(traindata.dataset)}")
    print(f"test data size: {len(testdata.dataset)}")
    _, new_acc = train_func.predict(model, testdata, per_point=True)
    print(f"model performance acc: {new_acc.sum()}")
    if per_point >= 0:
        scores = module.influences(np.arange(trainidx),  np.array([per_point])).cpu().numpy().reshape(-1)
    else:
        scores = module.influences(np.arange(trainidx), np.arange(testidx)).cpu().numpy().reshape(-1)
    return scores



class TrainFunc():
    def __init__(self, model, device, optimizer, scheduler, epochs, verbose):
        self._model = model
        self._device = device
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._epochs = epochs
        self._verbose = verbose

    def train_until_convergence(self, model, train_loader, test_loader, gradient_steps_between_eval):
        model = model.to(self._device)
        model.train()
        for epoch in range(self._epochs):
            for i, batch in enumerate(train_loader):
                self._optimizer.zero_grad()
                outputs = self._model(batch)
                loss = nn.CrossEntropyLoss()(outputs, batch['label'].to(self._device, dtype=torch.long))
                loss.backward()
                self._optimizer.step()
                if i % gradient_steps_between_eval == 0:
                    model.eval()
                    with torch.no_grad():
                        _, acc = self.predict(model, test_loader, per_point=False)
                        if self._verbose:
                            print(f"Epoch {epoch}, step {i}, acc: {acc.sum()}")
                    model.train()
            self._scheduler.step()
        return model, acc

    def predict(self, model, test_loader, per_point=False):
        # per_point: if True, return predicted label for each test point
        model = model.to(self._device)
        model.eval()
        acc = []
        for i, batch in enumerate(test_loader):
            outputs = model(batch)
            _, predicted = torch.max(outputs, -1)
            acc.append((predicted == batch['label'].to(self._device, dtype=torch.long)).cpu().numpy())
        acc = np.concatenate(acc)
        acc_val = acc.mean(-1)
        if per_point:
            return predicted, acc
        return acc_val
