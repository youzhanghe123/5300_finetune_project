import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from transformers import get_scheduler
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import time

from datasets import concatenate_datasets

from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd

import shutil
import os
import evaluate


# Step 1: Unzip the locally stored model file
# Specify the path to the ZIP file and the extraction folder
zip_file_path = "sft_T5.zip"
unzip_folder_path = "sft_T5"

# Unzip the file
if os.path.exists(unzip_folder_path):
    print(f"Directory {unzip_folder_path} already exists. Skipping extraction.")
else:
    shutil.unpack_archive(zip_file_path, unzip_folder_path)
    print(f"Extracted model to {unzip_folder_path}")

# Step 2: Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(unzip_folder_path)
tokenizer = T5Tokenizer.from_pretrained("distilbert-base-uncased")

# Step 3: Load the Data 
dataset = load_dataset("billsum", split="ca_test")
dataset = dataset.train_test_split(test_size=0.2)

# Split into train and test datasets
train_dataset = dataset["train"].select(range(900))  # Subset for demonstration (1000 samples)
test_dataset = dataset["test"].select(range(200))    # Subset for demonstration (200 samples)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# Step 4: Set up rouge evaluation
rouge = evaluate.load("rouge")

# Function to evaluate with ROUGE
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode the predictions and labels (token IDs) into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a list of predictions and references
    try:
        rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    except:
        print("error")
        print(decoded_preds)
        print(decoded_labels)

    # Extract relevant scores
    return  {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    predict_with_generate=True,  # Enables generation for ROUGE scoring
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,  # Loads the best model for evaluation
    metric_for_best_model="rouge2",  # Track ROUGE-2 for the best model

    logging_strategy="steps",  # Log by steps for continuous logs
    logging_steps=10,  # Set desired logging frequency
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

torch.cuda.empty_cache()

trainer.train()

trainer.evaluate()
