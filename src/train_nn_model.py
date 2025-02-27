import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from logging_config import logger
from utils import train_func_classification_nn, influence_at_checkpoints, validate_influence


parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, help="Number of epochs.", required=True)
parser.add_argument("--ckpt_list", type=int, nargs="+", help="List of checkpoints to save.", required=True)
parser.add_argument("--dir_data", type=str, help="Directory for datasets", required=True)
parser.add_argument("--dir_save_ckpt", type=str, help="Directory for saving checkpoints.", required=True)
parser.add_argument("--dir_save_tensors", type=str, help="Directory for saving tensors.", required=True)
parser.add_argument("--p_validate", type=float, help="Proportion of data to validate influences.", default=0.95)
parser.add_argument("--sample_size", type=int, help="Sample size for influence calculation.", required=True)
args = parser.parse_args()

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

mnist_train_loader = DataLoader(mnist_train_subset, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
for x, y in mnist_train_loader:
    print(x.shape)
    print(y.shape)
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
    

model = DigitNet()
model_name = type(model).__name__
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    "record_checkpoints": checkpoint_labels, 
    "dir_ckpt": dir_ckpt
}
checkpoint_filepaths, model_eval_obj = train_func_classification_nn(**train_func_kwargs)

influence_kwargs = {
    "traindata": mnist_train_loader, 
    "testdata": mnist_test_loader, 
    "model": model, 
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
    "criterion":  nn.CrossEntropyLoss(), 
    "checkpoints": checkpoint_filepaths, 
    "device": device, 
    "appro":"lissa"
}

# Calculate Influence Scores
logger.info("Calculating Train influences...")
mnist_influences = influence_at_checkpoints(**influence_kwargs)


# Validate Influences
logger.info("Validate influences...")
validate_influence_kwargs = {
    "influences": mnist_influences,
     "train_dataset": mnist_train_subset,
     "test_dataloader": mnist_test_loader,
     "model_class": DigitNet,
     "criterion": nn.CrossEntropyLoss(),
     "device": device,
     "checkpoint_filepaths": checkpoint_filepaths,
     "checkpoint_labels": checkpoint_labels,
     "p": args.p_validate
    }
validate_list, rand_list = validate_influence(**validate_influence_kwargs)


def save_info():
    logger.info("Saving train tensors...")
    dir_tensors = os.path.join(args.dir_save_tensors, "mnist")
    os.makedirs(dir_tensors, exist_ok=True)

    save_objs_dict = {
        "mnist_influences": mnist_influences,
        "model_eval_obj": model_eval_obj,
        "validate_list": validate_list,
        "rand_list": rand_list
    }
    torch.save(save_objs_dict, f"{dir_tensors}/accuracies_and_losses.pt")

save_info()
logger.info("Done!")



