import os
import argparse
import pathlib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, models
from tqdm import trange
from logging_config import logger
from utils import train_func_classification, influence_at_checkpoints, validate_influence


parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, help="Number of epochs.", required=True)
parser.add_argument("--ckpt_list", type=int, nargs="+", help="List of checkpoints to save.", required=True)
parser.add_argument("--dir_data", type=str, help="Directory for datasets", required=True)
parser.add_argument("--dir_save_ckpt", type=str, help="Directory for saving checkpoints.", required=True)
parser.add_argument("--dir_save_tensors", type=str, help="Directory for saving tensors.", required=True)
parser.add_argument("--dir_save_subset_indices", type=str, help="Directory for saving subset indices.", required=True)
parser.add_argument("--p_validate", type=float, help="Proportion of data to validate influences.", default=0.05)
args = parser.parse_args()

    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOGFISH_RAW_PATH = '../data/dogfish/dataset_dog-fish_train-900_test-300.npz'
DOGFISH_EMB_PATH = '../data/dogfish/dataset_dog-fish_embeds.npz'
dogfish_raw = np.load(DOGFISH_RAW_PATH, mmap_mode='r')

class DogfishDataset(Dataset):
    def __init__(self, X, Y, transforms=None):
        self.X = X
        self.Y = Y
        self.transforms = transforms
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = torch.tensor([self.Y[idx]])
        if self.transforms:
            sample = self.transforms(sample)
        return sample, label

# split: train or validation
def load_dogfish_examples(split, idxs):
    image = dogfish_raw[f"X_{split}"][idxs]
    label = dogfish_raw[f"Y_{split}"][idxs]
    return image, label

def dataloader_from_npz(path, type):
    data = np.load(path, "r")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.permute([2, 0, 1]))]) if type == "raw" else None
    train = DataLoader(DogfishDataset(data["X_train"], data["Y_train"], transforms=transform), batch_size=20, shuffle=True)
    test = DataLoader(DogfishDataset(data["X_test"], data["Y_test"], transforms=transform), batch_size=20, shuffle=False)
    return train, test

def load_inception_embeds():
    # load pretrained Inceptionv3
    if not pathlib.Path(DOGFISH_EMB_PATH).is_file():
        model = models.inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity()  # remove the last layer
        model.eval()
        model = model.to(DEVICE)

        embeds = {}
        raws = {}
        for split in ("train", "test"):
            n = 2 * (900 if (split == "train") else 300)

            X_embed, Y_embed = [], []
            X_raw, Y_raw = [], []
            for idx in trange(0, n, 20, desc=f"Embedding Dogfish ({split})"):
                batch_idxs = [idx + offset for offset in range(20)]
                images, labels = load_dogfish_examples(split, batch_idxs)
                images = torch.tensor(images, dtype=torch.float32, device=DEVICE)
                images = images.permute([0, 3, 1, 2]) # move channel to the second dimension
                X_raw.append(images)
                Y_raw.append(labels)
                
                # getting Inceptionv3 embedding
                with torch.no_grad():
                    h = model(images).cpu().numpy()

                X_embed.append(h)
                Y_embed.append(labels)

            embeds[f"X_{split}"] = np.concatenate(X_embed, axis=0).astype(np.float32)
            embeds[f"Y_{split}"] = np.concatenate(Y_embed, axis=0).astype(np.float32)
        
        np.savez(DOGFISH_EMB_PATH, **embeds)
        logger.info("Saved Dogfish embeddings!")
    logger.info("Loading Dogfish embeddings......")
    raw_train, raw_test = dataloader_from_npz(DOGFISH_RAW_PATH, "raw")
    embeds_train, embeds_test = dataloader_from_npz(DOGFISH_EMB_PATH, "embeds")
    return raw_train, raw_test, embeds_train, embeds_test

raw_train_loader, raw_validation_loader, embeds_train_loader, embeds_validation_loader = load_inception_embeds()
class V3FC(nn.Module):
    def __init__(self):
        super(V3FC, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.debug(f"Device: {device}")
model = V3FC().to(device)
model_name = type(model).__name__

# Train model and save check point at different stages
logger.info("Starting model training...")
num_epochs = args.num_epochs
checkpoint_labels = args.ckpt_list
logger.info(f"Checkpoint Labels: {checkpoint_labels}")
dir_ckpt = os.path.join(args.dir_save_ckpt, model_name)
os.makedirs(dir_ckpt, exist_ok=True)

train_func_kwargs = {
    "model": model,
    "criterion": nn.BCELoss(),
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
    "train_loader": embeds_train_loader,
    "validation_loader": embeds_validation_loader, 
    "epochs": num_epochs, 
    "predict_func": lambda pred: (pred > 0.5).int(),
    "device": device,
    "record_checkpoints": checkpoint_labels, 
    "dir_ckpt": dir_ckpt
}
checkpoint_filepaths, model_eval_obj = train_func_classification(**train_func_kwargs)

influence_kwargs = {
    "traindata": embeds_train_loader, 
    "testdata": embeds_validation_loader, 
    "model": model, 
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
    "criterion":  nn.BCELoss(), 
    "checkpoint_filepaths": checkpoint_filepaths, 
    "device": device, 
    "appro":"lissa"
}

# Calculate Influence Scores
logger.info("Calculating Train influences...")
dogfish_influences = influence_at_checkpoints(**influence_kwargs)


# Validate Influences
logger.info("Validate influences...")
embeds_data = np.load(DOGFISH_EMB_PATH, "r")
embeds_train_dataset = DogfishDataset(embeds_data["X_train"], embeds_data["Y_train"])
validate_influence_kwargs = {
    "influences": dogfish_influences,
     "train_dataset": embeds_train_dataset,
     "test_dataloader": embeds_validation_loader,
     "model_class": V3FC,
     "criterion": nn.BCELoss(),
     "device": device,
     "checkpoint_filepaths": checkpoint_filepaths,
     "checkpoint_labels": checkpoint_labels,
     "p": args.p_validate,
     "predict_func": lambda pred: (pred > 0.5).int()
    }
validate_list, rand_list = validate_influence(**validate_influence_kwargs)


def save_info():
    logger.info("Saving train tensors...")
    dir_tensors = os.path.join(args.dir_save_tensors, "dogfish")
    os.makedirs(dir_tensors, exist_ok=True)

    save_objs_dict = {
        "dogfish_influences": dogfish_influences,
        "model_eval_obj": model_eval_obj,
        "validate_list": validate_list,
        "rand_list": rand_list
    }
    torch.save(save_objs_dict, f"{dir_tensors}/accuracies_and_losses.pt")

save_info()
logger.info("Done!")



