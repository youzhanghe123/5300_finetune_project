import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer
from datasets import load_dataset
from transformers import get_scheduler
import time

from datasets import concatenate_datasets

from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd

import shutil
import os

# Step 1: Unzip the locally stored model file
# Specify the path to the ZIP file and the extraction folder
zip_file_path = "sft_distilbert.zip"
unzip_folder_path = "sft_distilbert"

# Unzip the file
if os.path.exists(unzip_folder_path):
    print(f"Directory {unzip_folder_path} already exists. Skipping extraction.")
else:
    shutil.unpack_archive(zip_file_path, unzip_folder_path)
    print(f"Extracted model to {unzip_folder_path}")

# Step 2: Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(unzip_folder_path)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load the IMDB dataset
dataset = load_dataset("imdb")


# Tokenize the IMDB dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

test_dataset = tokenized_datasets["test"]

test_dataset = test_dataset.shuffle(seed=42).select(  range(  len(test_dataset)//2+1, len(test_dataset) ) )

from torch.utils.data import DataLoader

def collate_fn(batch):
    # Convert lists to tensors directly and use padding
    inputs = {key: torch.tensor([example[key] for example in batch]) for key in tokenizer.model_input_names}
    labels = torch.tensor([example["label"] for example in batch])
    return inputs, labels

# Set up DataLoader with the new collate function
dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# Move model to GPU if available

# Store predictions and true labels
predictions = []
true_labels = []

# Predict on each batch
for batch in dataloader:
    inputs, labels = batch
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim=-1)

        # Append predictions and labels to lists
        predictions.extend(predicted_classes.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
from sklearn.metrics import precision_recall_fscore_support

# Calculate precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=1)
print ("after full supervised fine tune")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
