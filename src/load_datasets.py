import os
import pathlib
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import numpy as np
from logging_config import logger
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, models
from datasets import load_dataset
from transformers import BertModel, AutoTokenizer
from utils import set_seed

set_seed()

def load_mnist(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)) 
    ])
        
    train = datasets.MNIST(args.dir_data, train=True, download=True, transform=transform)
    test = datasets.MNIST(args.dir_data, train=False, download=True, transform=transform)
    logger.info("Loaded MNIST data!")

    sample_indices = np.random.choice(len(train), args.train_size, replace=False)
    train_subset = Subset(train, sample_indices)
    subset_dir = os.path.join(args.dir_save_subset_indices, "mnist")
    os.makedirs(subset_dir, exist_ok=True)
    torch.save(sample_indices, f"{subset_dir}/train_subset_indices.pt")
    logger.info("Saved subset indices!")

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    return train_subset, train_loader, test_loader

def load_dogfish(args, device):
    raw_path = os.path.join(args.dir_data, "dogfish/dataset_dog-fish_train-900_test-300.npz")
    emb_path = os.path.join(args.dir_data, "dogfish/dataset_dog-fish_embeds.npz")
    dogfish_raw = np.load(raw_path, mmap_mode='r')

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

    def load_inception_embeds(device):
        # load pretrained Inceptionv3
        if not pathlib.Path(emb_path).is_file():
            model = models.inception_v3(pretrained=True, transform_input=False)
            model.fc = nn.Identity()  # remove the last layer
            model.eval()
            model = model.to(device)

            embeds = {}
            raws = {}
            for split in ("train", "test"):
                n = 2 * (900 if (split == "train") else 300)

                X_embed, Y_embed = [], []
                X_raw, Y_raw = [], []
                for idx in trange(0, n, 20, desc=f"Embedding Dogfish ({split})"):
                    batch_idxs = [idx + offset for offset in range(20)]
                    images, labels = load_dogfish_examples(split, batch_idxs)
                    images = torch.tensor(images, dtype=torch.float32, device=device)
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
            
            np.savez(emb_path, **embeds)
            logger.info("Saved Dogfish embeddings!")
        logger.info("Loading Dogfish embeddings......")
        raw_train, raw_test = dataloader_from_npz(raw_path, "raw")
        embeds_train, embeds_test = dataloader_from_npz(emb_path, "embeds")
        return raw_train, raw_test, embeds_train, embeds_test
    
    raw_train_loader, raw_validation_loader, embeds_train_loader, embeds_validation_loader = load_inception_embeds(device)
    embeds_data = np.load(emb_path, "r")
    embeds_train_dataset = DogfishDataset(embeds_data["X_train"], embeds_data["Y_train"])
    return embeds_train_dataset, embeds_train_loader, embeds_validation_loader

def load_sst2(args, device):
    sst2_dataset = load_dataset("SetFit/sst2")
    logger.info("Loaded SST2 dataset!")
    language_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    embedding_train_path = os.path.join(args.dir_data, "sst-2/embedding_train.pt")
    embedding_val_path = os.path.join(args.dir_data, "sst-2/embedding_validation.pt")

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

    if pathlib.Path(embedding_train_path).is_file() and pathlib.Path(embedding_val_path).is_file():
        train_embeddings = torch.load(embedding_train_path, map_location=device)
        validation_embeddings = torch.load(embedding_val_path, map_location=device)
        logger.info("Loaded SST2 embeddings!")
        
    else:
        os.makedirs(os.path.join(args.dir_data, "sst-2"), exist_ok=True)
        train_embeddings = create_embeddings(type="train", dataset=sst2_dataset, model=language_model, out_filepath=embedding_train_path)
        validation_embeddings = create_embeddings(type="validation", dataset=sst2_dataset, model=language_model, out_filepath=embedding_val_path)

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
        
    train_dataset = SST2EmbedDataset(train_embeddings, sst2_dataset['train']['label'])
    sample_indices = np.random.choice(len(train_dataset), args.train_size, replace=False)
    train_subset = Subset(train_dataset, sample_indices)
    train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)

    validate_dataset = SST2EmbedDataset(validation_embeddings, sst2_dataset['validation']['label'])
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)
    os.makedirs(args.dir_save_subset_indices, exist_ok=True)
    torch.save(sample_indices, f"{args.dir_save_subset_indices}/sst2_embed_train_subset_indices.pt")
    logger.info("Saved subset indices!")
    return train_subset, train_dataloader, validate_dataloader