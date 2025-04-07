import argparse
import gc
import json
import os
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from preprocess import preprocess_raw_json_training_file_mc_ml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    zero_one_loss,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import wandb

load_dotenv()

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_name",
    type=str,
    default=None,
)
parser.add_argument(
    "--model_path",
    type=str,
)
parser.add_argument(
    "--epochs",
    type=int,
    default=80,
)
parser.add_argument(
    "--mode",
    type=str,
    default="online",
)
parser.add_argument("--loss_fn", type=str, default=None, required=False)
parser.add_argument("--focal_loss_gamma", type=float, default=2.0, required=False)
parser.add_argument("--gpu_index", type=str, default=None, required=False)

args = parser.parse_args()
run_name = args.run_name
model_path = args.model_path
epochs = args.epochs
mode = args.mode
loss_fn = args.loss_fn
focal_loss_gamma = args.focal_loss_gamma
gpu_index = args.gpu_index

if run_name is None:
    run_name = f"{model_path.split('/')[-1]}-{epochs}-epochs"


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_LOG_MODEL"]="end"

if gpu_index is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

TARGET_COLUMNS = [
    "Active Galactic Nuclei",
    "Binary Black Holes",
    "Binary Neutron Stars",
    "Binary Pulsars",
    "Burst",
    "Cataclysmic Variables",
    "Compact Binary Inspiral",
    "Continuous",
    "Cosmic Rays",
    "Exoplanets",
    "Fast Blue Optical Transients",
    "Fast Radio Bursts",
    "Gamma rays",
    "Gamma-ray Bursts",
    "Infrared",
    "Intermediate Mass",
    "Kilonovae",
    "Magnetars",
    "Neutrinos",
    "Neutron Star-Black Hole",
    "Novae",
    "Optical",
    "Pevatrons",
    "Pulsar Wind Nebulae",
    "Pulsars",
    "Radio",
    "Stellar Mass",
    "Stellar flares",
    "Stochastic",
    "SuperNovae",
    "Supermassive",
    "Supernova Remnants",
    "Ultraviolet",
    "White Dwarf Binaries",
    "X-rays",
    "non-TDAMM",
]
INPUT_COLUMNS = ["full_text"]
# MODEL_NAME = "nasa-impact/indus-sde-v0.1"
# MODEL_NAME = "nasa-impact/nasa-smd-ibm-v0.1"
MODEL_NAME = model_path

# AUTH_TOKEN = "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"
NUM_LABELS = len(TARGET_COLUMNS)
FREEZE_BASE = False

# Custom Dataset Class
class TDAMMDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class CustomTrainer(Trainer):
    def __init__(self, *args, focal_loss_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_gamma = focal_loss_gamma

    def focal_binary_cross_entropy(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        p = torch.sigmoid(logits)
        # Calculate focal loss
        p_t = torch.where(targets == 1, p, 1 - p)
        log_p_t = torch.log(torch.clamp(p_t, min=1e-4, max=1 - 1e-4))  # Cross Entropy
        loss = -((1 - p_t) ** self.focal_loss_gamma) * log_p_t
        return loss.mean()

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute focal loss
        loss = self.focal_binary_cross_entropy(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Metrics computation function
def compute_metrics(pred, threshold=0.5):
    labels = pred.label_ids
    logits = pred.predictions

    # Apply sigmoid to logits to get probabilities
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (probabilities > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="micro",
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
    )
    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _,
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
    )

    # Generate classification report
    report = classification_report(
        labels,
        predictions,
        target_names=TARGET_COLUMNS,
        output_dict=True,
    )

    return {
        "accuracy": accuracy,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "hamming_loss": hamming_loss(labels, predictions),
        "zero_one_loss": zero_one_loss(labels, predictions),
        "jaccard_score": jaccard_score(labels, predictions, average="weighted"),
    }


with open("tdamm_training_data_multi_labelled.json", "r") as file:
    data = json.load(file)

df = preprocess_raw_json_training_file_mc_ml(data)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(
    list(train_df[INPUT_COLUMNS[0]].values),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)

test_encodings = tokenizer(
    list(test_df[INPUT_COLUMNS[0]].values),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create datasets
train_dataset = TDAMMDataset(
    train_encodings,
    torch.tensor(train_df[TARGET_COLUMNS].values, dtype=torch.float),
)

test_dataset = TDAMMDataset(
    test_encodings,
    torch.tensor(test_df[TARGET_COLUMNS].values, dtype=torch.float),
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    # token=AUTH_TOKEN,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
)

if FREEZE_BASE:
    for param in model.base_model.parameters():
        param.requires_grad = False

model = model.to(device)
# Initialize wandb
wandb.init(
    project="tdamm-classification",
    name=run_name,
    mode=mode,
)
# wandb.watch(model, log="all")
wandb.config.update(
    {
        "model_name": MODEL_NAME,
        "freeze_base": FREEZE_BASE,
        "num_labels": NUM_LABELS,
        "target_columns": TARGET_COLUMNS,
        "input_columns": INPUT_COLUMNS,
        "data_size": len(df),
        "loss_fn": loss_fn,
        "focal_loss_gamma": focal_loss_gamma,
        "gpu_index": gpu_index,
    },
)

# Training arguments without early stopping parameters
training_args = TrainingArguments(
    output_dir="./tdamm_results_v4",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=epochs,
    bf16=True,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    # warmup_steps=500,
    # gradient_checkpointing=True, # slows down the backward pass although saving memory
    logging_steps=500,
    save_total_limit=2,
    metric_for_best_model="precision_weighted",
    greater_is_better=True,
    optim="adamw_torch",
    # use_cpu=True,
)

# Create early stopping callback
# early_stopping = EarlyStoppingCallback(
#     early_stopping_patience=3,
#     early_stopping_threshold=0.01
# )
trainer_config = dict(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    # callbacks=[early_stopping]
)

if loss_fn == "focal_loss":
    trainer = CustomTrainer(
        **trainer_config,
        focal_loss_gamma=focal_loss_gamma,
    )
else:
    trainer = Trainer(
        **trainer_config,
    )

# Train model
trainer.train()
eval_results = trainer.evaluate()
wandb.log({f"Evaluation {run_name} epochs": eval_results})

eval_report_map = dict(
    train=trainer.evaluate(train_dataset),
    test=trainer.evaluate(test_dataset),
)

os.makedirs(f"eval_backup/", exist_ok=True)
os.makedirs(f"eval_backup/{run_name}/", exist_ok=True)

train_eval_ = pd.DataFrame(eval_report_map["train"]["eval_classification_report"]).T
test_eval_ = pd.DataFrame(eval_report_map["test"]["eval_classification_report"]).T

train_eval_.to_csv(f"eval_backup/{run_name}/train_eval.csv")
test_eval_.to_csv(f"eval_backup/{run_name}/test_eval.csv")

eval_dct = {}
for eval_type, eval_report in eval_report_map.copy().items():
    classification_report_df = pd.DataFrame(
        eval_report.get("eval_classification_report"),
    )
    print(f">>> {eval_type}")
    print(eval_report)
    _ = eval_report.pop("eval_classification_report")
    eval_dct[eval_type] = eval_report
    wandb.log(
        {
            f"{eval_type}_classification_report": wandb.Table(
                dataframe=classification_report_df,
            ),
        },
    )
    print(f"{eval_type} <<< ")

wandb.log(eval_dct)
wandb.finish()
