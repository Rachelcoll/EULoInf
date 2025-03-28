import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from logging_config import logger
from utils import set_seed, train_func_classification, validate_influence
from calc_influences.influence_function import influence_at_checkpoints_if
from calc_influences.tracin import influence_at_checkpoints_tracin
from calc_influences.tmc_shapley import influence_at_checkpoints_tmc_shapley


parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, help="Number of epochs.", required=True)
parser.add_argument("--ckpt_list", type=int, nargs="+", help="List of checkpoints to save.", required=True)
parser.add_argument("--dir_data", type=str, help="Directory for datasets", required=True)
parser.add_argument("--dir_save_ckpt", type=str, help="Directory for saving checkpoints.", required=True)
parser.add_argument("--dir_save_tensors", type=str, help="Directory for saving tensors.", required=True)
parser.add_argument("--dir_save_subset_indices", type=str, help="Directory for saving subset indices.", required=True)
parser.add_argument("--p_validate", type=float, help="Proportion of data to validate influences.", default=0.05)
parser.add_argument("--sample_size", type=int, help="Sample size for influence calculation.", required=True)
args = parser.parse_args()
set_seed()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)) 
])
    
mnist_train = datasets.MNIST(args.dir_data, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(args.dir_data, train=False, download=True, transform=transform)
logger.info("Loaded MNIST data!")

sample_size = args.sample_size
sample_indices = np.random.choice(len(mnist_train), sample_size, replace=False)
mnist_train_subset = Subset(mnist_train, sample_indices)
os.makedirs(args.dir_save_subset_indices, exist_ok=True)
torch.save(sample_indices, f"{args.dir_save_subset_indices}/mnist_train_subset_indices.pt")
logger.info("Saved subset indices!")

mnist_train_loader = DataLoader(mnist_train_subset, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
for x, y in mnist_train_loader:
    logger.debug(f"{x.shape}")
    logger.debug(f"{y.shape}")
    break

class DigitNet(nn.Module):
    def __init__(self, input_dimensions=784, num_classes=10): 
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dimensions, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=-1)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def to(self, device):
        super().to(device)
        self.device = device
        return self
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.debug(f"Device: {device}")
model = DigitNet().to(device)
model_name = type(model).__name__

# Train model and save check point at different stages
logger.info("Starting model training...")
num_epochs = args.num_epochs
checkpoint_labels = args.ckpt_list # checkpoint_labels = np.array([3, 5, 7, 10, 15, 20, 40, 60, 80, 100])
logger.info(f"Checkpoint Labels: {checkpoint_labels}")
dir_ckpt = os.path.join(args.dir_save_ckpt, model_name)
os.makedirs(dir_ckpt, exist_ok=True)

train_func_kwargs = {
    "model": model,
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
    "train_loader": mnist_train_loader,
    "validation_loader": mnist_test_loader, 
    "epochs": num_epochs, 
    "predict_func": lambda pred: torch.argmax(pred, dim=-1),
    "device": device,
    "record_checkpoints": checkpoint_labels, 
    "dir_ckpt": dir_ckpt
}
checkpoint_filepaths, model_eval_obj = train_func_classification(**train_func_kwargs)

# Method 1: Influence Function calculation
influence_kwargs = {
    "traindata": mnist_train_loader, 
    "testdata": mnist_test_loader, 
    "model": model, 
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
    "criterion":  nn.CrossEntropyLoss(), 
    "checkpoint_filepaths": checkpoint_filepaths, 
    "device": device, 
    "appro":"lissa"
}
logger.info("Calculating Train influences using influence function...")
mnist_influences_if = influence_at_checkpoints_if(**influence_kwargs)

# Method 2: Tracin calculation
influence_kwargs = {
    "traindata": mnist_train_loader, 
    "testdata": mnist_test_loader, 
    "model": model, 
    "criterion":  nn.CrossEntropyLoss(reduction="none"), 
    "checkpoint_filepaths": checkpoint_filepaths, 
}
logger.info("Calculating Train influences using Tracin...")
mnist_influences_tracin = influence_at_checkpoints_tracin(**influence_kwargs)

# Method 3: TMC Shapely calculation
influence_kwargs = {
    "traindata": mnist_train_loader,
    "testdata": mnist_test_loader,
    "model": model,
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
    "checkpoint_filepaths": checkpoint_filepaths
}
mnist_influences_tmc_shapley = influence_at_checkpoints_tmc_shapley(**influence_kwargs)

# Validate Influences
validate_influence_kwargs = {
    "train_dataset": mnist_train_subset,
    "test_dataloader": mnist_test_loader,
    "model_class": DigitNet,
    "criterion": nn.CrossEntropyLoss(),
    "device": device,
    "checkpoint_filepaths": checkpoint_filepaths,
    "checkpoint_labels": checkpoint_labels,
    "p": args.p_validate,
    "predict_func": lambda pred: torch.argmax(pred, dim=-1)
    }
logger.info("Validate influences from Influence Function...")
validate_list_if, rand_list_if = validate_influence(influences=mnist_influences_if, **validate_influence_kwargs)
logger.info("Validate influences from Tracin...")
validate_list_tracin, rand_list_tracin = validate_influence(influences=mnist_influences_tracin, **validate_influence_kwargs)
logger.info("Validate influences from TMC Shapley...")
validate_list_tmc_shapley, rand_list_tmc_shapley = validate_influence(influences=mnist_influences_tmc_shapley, **validate_influence_kwargs)


def save_info():
    logger.info("Saving train tensors...")
    dir_tensors = os.path.join(args.dir_save_tensors, "mnist")
    os.makedirs(dir_tensors, exist_ok=True)

    save_objs_dict = {
        "model_eval_obj": model_eval_obj,
        "mnist_influences_if": mnist_influences_if,
        "validate_list_if": validate_list_if,
        "rand_list_if": rand_list_if,
        "mnist_influences_tracin": mnist_influences_tracin,
        "validate_list_tracin": validate_list_tracin,
        "rand_list_tracin": rand_list_tracin,
        "mnist_influences_tmc_shapley": mnist_influences_tmc_shapley,
        "validate_list_tmc_shapley": validate_list_tmc_shapley,
        "rand_list_tmc_shapley": rand_list_tmc_shapley
    }
    torch.save(save_objs_dict, f"{dir_tensors}/accuracies_and_losses.pt")

save_info()
logger.info("Done!")



