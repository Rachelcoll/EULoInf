import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pathlib
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
from datasets import load_dataset
from transformers import BertModel, AutoTokenizer
from logging_config import logger
from utils import set_seed, train_func_classification, validate_influence
from calc_influences.influence_function import influence_at_checkpoints_if
from calc_influences.tracin import influence_at_checkpoints_tracin_updated


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

sst2_dataset = load_dataset("SetFit/sst2")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.debug(f"Device: {device}")
language_model = BertModel.from_pretrained("bert-base-uncased").to(device)

logger.info("Loaded SST2 dataset!")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

EMBEDDING_TRAIN_SST2 = "../data/sst-2/embedding_train.pt"
EMBEDDING_VALIDATION_SST2 = "../data/sst-2/embedding_validation.pt"

def create_embeddings(type, dataset, model, out_filepath):
    data_texts = dataset[type]["text"]
    sst2_embeddings = []
    max_length = max([len(tokenizer.tokenize(text)) for text in dataset[type]['text']])
    for text in tqdm(data_texts, desc="Embedding train SST2", total=len(data_texts)):
        tokenized_text = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt", padding="max_length")
        tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
        with torch.no_grad():
            embedding = model(**tokenized_text).last_hidden_state.cpu()
        sst2_embeddings.append(embedding)
        del tokenized_text
        del embedding
        torch.cuda.empty_cache()
    torch.save(sst2_embeddings, out_filepath)
    return sst2_embeddings


if pathlib.Path(EMBEDDING_TRAIN_SST2).is_file() and pathlib.Path(EMBEDDING_VALIDATION_SST2).is_file():
    sst2_train_embeddings = torch.load(EMBEDDING_TRAIN_SST2, map_location=device)
    sst2_validation_embeddings = torch.load(EMBEDDING_VALIDATION_SST2, map_location=device)
    logger.info("Loaded SST2 embeddings!")
    
else:
    os.makedirs("../data/sst-2", exist_ok=True)
    sst2_train_embeddings = create_embeddings(type="train", dataset=sst2_dataset, model=language_model, out_filepath=EMBEDDING_TRAIN_SST2)
    sst2_validation_embeddings = create_embeddings(type="validation", dataset=sst2_dataset, model=language_model, out_filepath=EMBEDDING_VALIDATION_SST2)

class SST2EmbedDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx])
        label = torch.tensor(self.labels[idx])
        return embedding, label
    
sst2_embed_train_dataset = SST2EmbedDataset(sst2_train_embeddings, sst2_dataset['train']['label'])
sample_size = args.sample_size
sample_indices = np.random.choice(len(sst2_embed_train_dataset), sample_size, replace=False)
sst2_embed_train_subset = Subset(sst2_embed_train_dataset, sample_indices)
sst2_embed_train_dataloader = DataLoader(sst2_embed_train_subset, batch_size=64, shuffle=True)

sst2_embed_validate_dataset = SST2EmbedDataset(sst2_validation_embeddings, sst2_dataset['validation']['label'])
sst2_embed_validate_dataloader = DataLoader(sst2_embed_validate_dataset, batch_size=64, shuffle=False)
os.makedirs(args.dir_save_subset_indices, exist_ok=True)
torch.save(sample_indices, f"{args.dir_save_subset_indices}/sst2_embed_train_subset_indices.pt")
logger.info("Saved subset indices!")


class TextCNN(nn.Module):
    def __init__(self, embed_dim=768, num_classes=2, kernel_sizes=[2, 3, 4], num_filters=2):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        conv_outputs = []
        for conv in self.convs:
            conv_output = F.relu(conv(x)) # [batch size, num_filters, seq_len - kernel_size + 1, 1]
            conv_output_pool = F.max_pool2d(conv_output, (conv_output.shape[2], 1))
            conv_outputs.append(conv_output_pool.squeeze(3).squeeze(2))
        output = torch.cat(conv_outputs, dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        return output
    
    def to(self, device):
        super().to(device)
        self.device = device
        return self

model = TextCNN().to(device)
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
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
    "train_loader": sst2_embed_train_dataloader,
    "validation_loader": sst2_embed_validate_dataloader, 
    "epochs": num_epochs, 
    "predict_func": lambda pred: torch.argmax(pred, dim=-1),
    "device": device,
    "record_checkpoints": checkpoint_labels, 
    "dir_ckpt": dir_ckpt
}
checkpoint_filepaths, model_eval_obj = train_func_classification(**train_func_kwargs)

# Method 1: Influence Function calculation
influence_kwargs = {
    "traindata": sst2_embed_train_dataloader, 
    "testdata": sst2_embed_validate_dataloader, 
    "model": model, 
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
    "criterion":  nn.CrossEntropyLoss(), 
    "checkpoint_filepaths": checkpoint_filepaths, 
    "device": device, 
    "appro":"lissa"
}
logger.info("Calculating Train influences using influence function...")
sst2_influences_if = influence_at_checkpoints_if(**influence_kwargs)
import pickle
pickle.dump(sst2_influences_if, open("/home/nicole/Workspace/data_valuation/influences/sst2/sst2_influences_if.pkl", "wb"))


# Method 2: Tracin calculation
influence_kwargs = {
    "traindata": sst2_embed_train_dataloader, 
    "testdata": sst2_embed_validate_dataloader, 
    "model": model, 
    "criterion":  nn.CrossEntropyLoss(reduction="none"), 
    "checkpoint_filepaths": checkpoint_filepaths, 
}
logger.info("Calculating Train influences using Tracin...")
sst2_influences_tracin = influence_at_checkpoints_tracin_updated(**influence_kwargs)
pickle.dump(sst2_influences_tracin, open("/home/nicole/Workspace/data_valuation/influences/sst2/sst2_influences_tracin.pkl", "wb"))


# Validate Influences
validate_influence_kwargs = {
     "train_dataset": sst2_embed_train_subset,
     "test_dataloader": sst2_embed_validate_dataloader,
     "model_class": TextCNN,
     "criterion": nn.CrossEntropyLoss(),
     "device": device,
     "checkpoint_filepaths": checkpoint_filepaths,
     "checkpoint_labels": checkpoint_labels,
     "p": args.p_validate,
     "predict_func": lambda pred: torch.argmax(pred, dim=-1)
    }
logger.info("Validate influences from Influence Function...")
validate_list_if, rand_list_if = validate_influence(influences=sst2_influences_if, **validate_influence_kwargs)
logger.info("Validate influences from Tracin...")
validate_list_tracin, rand_list_tracin = validate_influence(influences=sst2_influences_tracin, **validate_influence_kwargs)


def save_info():
    logger.info("Saving train tensors...")
    dir_tensors = os.path.join(args.dir_save_tensors, "sst2")
    os.makedirs(dir_tensors, exist_ok=True)

    save_objs_dict = {
        "model_eval_obj": model_eval_obj,
        "sst2_influences_if": sst2_influences_if,
        "validate_list_if": validate_list_if,
        "rand_list_if": rand_list_if,
        "sst2_influences_tracin": sst2_influences_tracin,
        "validate_list_tracin": validate_list_tracin,
        "rand_list_tracin": rand_list_tracin
    }
    torch.save(save_objs_dict, f"{dir_tensors}/accuracies_and_losses.pt")

save_info()
logger.info("Done!")



