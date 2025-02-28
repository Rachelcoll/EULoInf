import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pathlib
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from datasets import load_dataset
from transformers import BertModel, AutoTokenizer
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
parser.add_argument("--sample_size", type=int, help="Sample size for influence calculation.", required=True)
args = parser.parse_args()

sst2_dataset = load_dataset("SetFit/sst2")
model = BertModel.from_pretrained("bert-base-uncased")

logger.info("Loaded SST2 data!")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_text_lengths = [len(tokenizer.tokenize(text)) for text in sst2_dataset['train']['text']]
validation_text_lengths = [len(tokenizer.tokenize(text)) for text in sst2_dataset['validation']['text']]
max_train_length = max(train_text_lengths)
max_validation_length = max(validation_text_lengths)


EMBEDDING_TRAIN_SST2 = "../data/sst-2/embedding_train.pt"
EMBEDDING_VALIDATION_SST2 = "../data/sst-2/embedding_validation.pt"
train_texts = sst2_dataset['train']['text']
validation_texts = sst2_dataset['validation']['text']
sst2_train_embeddings = []
sst2_validation_embeddings = []
if pathlib.Path(EMBEDDING_TRAIN_SST2).is_file() and pathlib.Path(EMBEDDING_VALIDATION_SST2).is_file():
    sst2_train_embeddings = torch.load(EMBEDDING_TRAIN_SST2)
    sst2_validation_embeddings = torch.load(EMBEDDING_VALIDATION_SST2)
    logger.info("Loaded SST2 embeddings!")
    
else:
    os.makedirs("../data/sst-2", exist_ok=True)
    for text in tqdm(train_texts, desc="Embedding train SST2", total=len(train_texts)):
        tokenized_text = tokenizer(text, truncation=True, max_length=max_train_length, return_tensors="pt", padding="max_length")
        embedding = model(**tokenized_text).last_hidden_state
        sst2_train_embeddings.append(embedding)
    torch.save(sst2_train_embeddings, EMBEDDING_TRAIN_SST2)
        
    for text in tqdm(validation_texts, desc="Embedding train SST2", total=len(validation_texts)):
        tokenized_text = tokenizer(text, truncation=True, max_length=max_validation_length, return_tensors="pt", padding="max_length")
        embedding = model(**tokenized_text).last_hidden_state
        sst2_validation_embeddings.append(embedding)    
    torch.save(sst2_validation_embeddings, EMBEDDING_VALIDATION_SST2)
    
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

model = TextCNN()
model_name = type(model).__name__
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    "record_checkpoints": checkpoint_labels, 
    "dir_ckpt": dir_ckpt
}
checkpoint_filepaths, model_eval_obj = train_func_classification(**train_func_kwargs)

influence_kwargs = {
    "traindata": sst2_embed_train_dataloader, 
    "testdata": sst2_embed_validate_dataloader, 
    "model": model, 
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.0001),
    "criterion":  nn.CrossEntropyLoss(), 
    "checkpoints": checkpoint_filepaths, 
    "device": device, 
    "appro":"lissa"
}

# Calculate Influence Scores
logger.info("Calculating Train influences...")
sst2_influences = influence_at_checkpoints(**influence_kwargs)


# Validate Influences
logger.info("Validate influences...")
validate_influence_kwargs = {
    "influences": sst2_influences,
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
validate_list, rand_list = validate_influence(**validate_influence_kwargs)


def save_info():
    logger.info("Saving train tensors...")
    dir_tensors = os.path.join(args.dir_save_tensors, "sst2")
    os.makedirs(dir_tensors, exist_ok=True)

    save_objs_dict = {
        "sst2_influences": sst2_influences,
        "model_eval_obj": model_eval_obj,
        "validate_list": validate_list,
        "rand_list": rand_list
    }
    torch.save(save_objs_dict, f"{dir_tensors}/accuracies_and_losses.pt")

save_info()
logger.info("Done!")



