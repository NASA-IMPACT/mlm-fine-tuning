import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
import wandb
from evals import generate_eval_table
from preprocess_data import preprocess_dataset
from transformers import (
    AutoModelForMaskedLM,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from utils import printd


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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    default="config.json",
    help="Path to config file",
)
parser.add_argument("--gpu_index_to_use", type=str, default=None, required=False)
parser.add_argument(
    "--nrows",
    type=int,
    default=None,
    help="Limit the dataset size with n rows",
)

# Parse arguments
args = parser.parse_args()
config_path = args.config_path
gpu_index_to_use = args.gpu_index_to_use
n_rows = args.nrows

with open("config.json", "r") as file:
    config = json.load(file)
config["termina_args"] = vars(args)

current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H-%M-%S")


model_output_dir = os.path.join(
    config.get("output").get("dir"),
    str(formatted_datetime),
)
os.makedirs(model_output_dir, exist_ok=True)
file = open(os.path.join(model_output_dir, config.get("output").get("log")), "w")

if gpu_index_to_use is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index_to_use

printd("*" * 10 + "GPUs" + "*" * 10, file=file)
for i in range(torch.cuda.device_count()):
    printd(torch.cuda.get_device_properties(i).name, file=file)
printd("*" * 30, file=file)

assert os.getenv("WANDB_LOG_MODEL") == "end"
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="mlm-fine-tuning")

if __name__ == "__main__":
    printd("*" * 10 + "Started DataPreprocessing" + "*" * 10, file=file)
    lm_dataset, tokenizer, data_collator = preprocess_dataset(config.get("input"))
    model = AutoModelForMaskedLM.from_pretrained(
        config.get("input").get("model").get("hf"),
    )

    training_args = TrainingArguments(
        output_dir=f"{config.get('output').get('model_backups_path')}timestamp_{formatted_datetime}/{config.get('input').get('model').get('hf')}/",
        **config.get("TrainingArguments"),
    )
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.get("additional_training_config").get(
            "training_patience",
        ),
    )
    trainer_config = dict(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        callbacks=[early_stopping],
        tokenizer=tokenizer,
    )
    trainer = Trainer(**trainer_config)

    wandb.watch(model, log="all")
    # update the config to wandb
    custom_cfg = dict(
        **config,
        train_size=sum([len(i) for i in lm_dataset["train"]["input_ids"]]),
        val_size=sum([len(i) for i in lm_dataset["validation"]["input_ids"]]),
        test_size=sum([len(i) for i in lm_dataset["test"]["input_ids"]]),
    )
    custom_cfg["data_size"] = (
        custom_cfg["train_size"] + custom_cfg["val_size"] + custom_cfg["test_size"]
    )
    wandb.config.update(custom_cfg)

    printd("*" * 10 + "Started Training" + "*" * 10, file=file)
    printd(lm_dataset, file=file)
    trainer.train()

    model_save_loc = os.path.join(
        model_output_dir,
        config.get("output").get("final_model_path"),
    )
    os.makedirs(model_save_loc, exist_ok=True)
    model.save_pretrained(model_save_loc)
    tokenizer.save_pretrained(model_save_loc)

    eval_report_df = generate_eval_table(trainer, lm_dataset)
    wandb.log({f"eval_report": wandb.Table(dataframe=eval_report_df)})
    #
    wandb.finish()

file.close()
