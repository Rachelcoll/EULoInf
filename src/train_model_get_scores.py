import os
import argparse
import torch
import torch.nn as nn

from utils import set_seed, incremental_save, train_func_classification, validate_influence
from load_datasets import load_mnist, load_dogfish, load_sst2
from load_models import DigitNet, V3FC, TextCNN
from calc_influences.influence_function import influence_at_checkpoints_if
from calc_influences.tracin import influence_at_checkpoints_tracin
from calc_influences.tmc_shapley import influence_at_checkpoints_tmc_shapley
from datetime import datetime
from logging_config import logger

parser = argparse.ArgumentParser()
# Dataset related
parser.add_argument("--data_name", type=str, help="Name of dataset.", required=True)
parser.add_argument("--model_name", type=str, help="Name of model to use.", required=True)
parser.add_argument("--dir_data", type=str, help="Directory for datasets", required=True)
parser.add_argument("--train_size", type=int, help="Train samples.")
parser.add_argument("--batch_size", type=int, help="Batch size.")
parser.add_argument("--dir_save_subset_indices", type=str, help="Directory for saving subset indices.", required=True)

# Training related
parser.add_argument("--num_epochs", type=int, help="Number of epochs.", required=True)
parser.add_argument("--ckpt_list", type=int, nargs="+", help="List of checkpoints to save.", required=True)
parser.add_argument("--dir_save_ckpt", type=str, help="Directory for saving checkpoints.", required=True)

# Valuation related
parser.add_argument("--scoring_methods", type=str, nargs="+", help="List of valuation methods to do.", required=True)
parser.add_argument("--dir_save_tensors", type=str, help="Directory for saving tensors.", required=True)
parser.add_argument("--p_validate", type=float, help="Proportion of data to validate influences.", default=0.05)

args = parser.parse_args()
set_seed()

logger.debug(f"{args}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading dataset
dataset_options = {
    "mnist": {"data" : load_mnist, "args": [args]},
    "dogfish": {"data": load_dogfish, "args": [args, device]},
    "sst2": {"data": load_sst2, "args": [args, device]}
}
load_dataset_func = dataset_options[args.data_name]["data"]
load_dataset_args = dataset_options[args.data_name]["args"]
train_dataset, train_loader, test_loader = load_dataset_func(*load_dataset_args)

# Loading model
model_options = {
    "digitnet": {"model": DigitNet, "predict_func": lambda pred: torch.argmax(pred, dim=-1), "criterion": nn.CrossEntropyLoss()},
    "v3fc": {"model": V3FC, "predict_func": lambda pred: (pred > 0.5).int(), "criterion": nn.BCELoss()},
    "textcnn": {"model": TextCNN, "predict_func": lambda pred: torch.argmax(pred, dim=-1), "criterion": nn.CrossEntropyLoss()}
}
model = model_options[args.model_name]["model"]().to(device)
predict_func = model_options[args.model_name]["predict_func"]
criterion = model_options[args.model_name]["criterion"]
model_name = type(model).__name__

logger.info("Starting model training...")
dir_ckpt = os.path.join(args.dir_save_ckpt, model_name)
os.makedirs(dir_ckpt, exist_ok=True)

train_func_kwargs = {
    "model": model,
    "criterion": criterion,
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
    "train_loader": train_loader,
    "validation_loader": test_loader, 
    "epochs": args.num_epochs, 
    "predict_func": predict_func,
    "device": device,
    "record_checkpoints": args.ckpt_list, 
    "dir_ckpt": dir_ckpt
}
checkpoint_filepaths, model_eval_obj = train_func_classification(**train_func_kwargs)
save_objs_dict = {"model_eval_obj": model_eval_obj}
incremental_save(args, save_objs_dict, step_name="training")

# INFLUENCE calculations
if "if" in args.scoring_methods:
    # Option 1: Influence Function calculation
    influence_kwargs = {
        "traindata": train_loader, 
        "testdata": test_loader, 
        "model": model, 
        "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
        "criterion":  criterion, 
        "checkpoint_filepaths": checkpoint_filepaths, 
        "device": device, 
        "appro":"lissa"
    }
    logger.info("Calculating train influences using influence function...")
    influences_if = influence_at_checkpoints_if(**influence_kwargs)
    save_objs_dict.update({"influences_if": influences_if})
    incremental_save(args, save_objs_dict, step_name="if")


if "tracin" in args.scoring_methods:
    # Option 2: Tracin calculation
    influence_kwargs = {
        "traindata": train_loader, 
        "testdata": test_loader, 
        "model": model, 
        "criterion":  criterion, 
        "checkpoint_filepaths": checkpoint_filepaths, 
    }
    logger.info("Calculating train influences using Tracin...")
    influences_tracin = influence_at_checkpoints_tracin(**influence_kwargs)
    save_objs_dict.update({"influences_tracin": influences_tracin})
    incremental_save(args, save_objs_dict, step_name="tracin")

if "tmc_shapley" in args.scoring_methods:
    # Option 3: TMC Shapely calculation
    influence_kwargs = {
        "traindata": train_loader,
        "testdata": test_loader,
        "model": model,
        "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
        "predict_func": predict_func,
        "checkpoint_filepaths": checkpoint_filepaths,
        "device": device
    }
    logger.info("Calculating train influences using TMC Shapley...")
    influences_tmc_shapley = influence_at_checkpoints_tmc_shapley(**influence_kwargs)
    save_objs_dict.update({"influences_tmc_shapley": influences_tmc_shapley})
    incremental_save(args, save_objs_dict, step_name="tmc_shapley")

# Validate influences
validate_influence_kwargs = {
    "train_dataset": train_dataset,
    "test_dataloader": test_loader,
    "model_class": model_options[args.model_name]["model"],
    "criterion": criterion,
    "device": device,
    "checkpoint_filepaths": checkpoint_filepaths,
    "checkpoint_labels": args.ckpt_list,
    "p": args.p_validate,
    "predict_func": predict_func
    }
for method in args.scoring_methods:
    logger.info(f"Validate influences from {method}...")
    influences = save_objs_dict[f"influences_{method}"]
    validate_list, rand_list = validate_influence(influences=influences, **validate_influence_kwargs)
    save_objs_dict.update({f"validate_list_{method}": validate_list, f"rand_list_{method}": rand_list})
    incremental_save(args, save_objs_dict, step_name=f"VALIDATE {method}")

# SAVE DATA
time_suffix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger.info("Saving all tensors...")
dir_tensors = os.path.join(args.dir_save_tensors, f"{args.data_name}_{time_suffix}")
os.makedirs(dir_tensors, exist_ok=True)
torch.save(save_objs_dict, f"{dir_tensors}/accuracies_and_losses.pt")
logger.info("Done!")





