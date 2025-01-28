import datetime
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from accelerate.commands.config.update import description
from transformers import (
    AutoModelForMaskedLM,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import wandb


def set_seed(seed=42) -> None:
    """Set all seeds to make results reproducible (deterministic mode).
    When seed is a false-y value or not supplied, disables deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

from dotenv import load_dotenv

load_dotenv()

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)


def load_model_from_wandb(model_path):
    run = wandb.init()
    artifact = run.use_artifact(model_path, type="model")
    artifact_dir = artifact.download()
    model = AutoModelForMaskedLM.from_pretrained(artifact_dir)
    return model, artifact_dir


def load_model_from_local(model_path):
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    return model, model_path


def load_model(model_path, use_wandb=True):
    if use_wandb:
        return load_model_from_wandb(model_path)
    else:
        return load_model_from_local(model_path)


# checking if two hugging face models are same or not
model_from_wandb, model_from_wandb_path = load_model_from_wandb(
    "nasa-impact/mlm-fine-tuning/model-ytxjrbhy:v1",
)
model_from_local, model_from_local_path = load_model_from_local(
    "/rhome/sawale/indus_traning/mlm-fine-tuning/mlm/tmp/models/timestamp_20241216_18-41-39/nasa-impact/nasa-smd-ibm-v0.1/checkpoint-23000",
)
# model_from_local, model_from_local_path = load_model_from_local("/rhome/sawale/indus_traning/mlm-fine-tuning/mlm/model_outputs/20241211_15-10-59/model")


# Compare the state_dicts
are_models_identical = all(
    torch.equal(param1, param2)
    for param1, param2 in zip(
        model_from_wandb.state_dict().values(),
        model_from_local.state_dict().values(),
    )
)

print(f"Are the models identical? {are_models_identical}")

from preprocess_data import preprocess_dataset

with open("config.json", "r") as file:
    config = json.load(file)

data_src = "local"
n_rows = None
lm_dataset, tokenizer, data_collator = preprocess_dataset(
    config.get("input"),
    data_src,
    n_rows,
)

from utils import generate_inference

test_inference_df = generate_inference(
    lm_dataset["test"],
    tokenizer,
    model_from_local_path,
    top_k=config.get("output").get("inference").get("top_k"),
    n_predictions=None,
)

train_inference_df = generate_inference(
    lm_dataset["train"],
    tokenizer,
    model_from_local_path,
    top_k=config.get("output").get("inference").get("top_k"),
    n_predictions=None,
)
train_inference_df.to_parquet("tmp/inferences/train_inference.parquet")
test_inference_df.to_parquet("tmp/inferences/test_inference.parquet")
