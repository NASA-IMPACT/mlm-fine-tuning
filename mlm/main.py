import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
from dotenv import load_dotenv

# from accelerate.commands.config.update import description
from evals import generate_eval_table
from pefts import get_model
from preprocess_data import preprocess_dataset, preprocess_dataset_with_kw_masking
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from utils import generate_inference, printd

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

load_dotenv()

base_path = os.path.dirname(__file__)  # Path of the current script
print(base_path)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    default="config_new_data.json",
    help="Path to config file",
)
parser.add_argument(
    "--data_src",
    type=str,
    default="local",
    required=False,
    help="local or hf",
)
parser.add_argument("--gpu_index_to_use", type=str, default=None, required=False)
parser.add_argument(
    "--train_techs",
    type=str,
    default=[],
    required=False,
    nargs="*",
    help="to use lora, quant",
)
parser.add_argument(
    "--nrows",
    type=int,
    default=None,
    help="Limit the dataset size with n rows",
)
parser.add_argument(
    "--resume_checkpoint_path",
    type=str,
    default=None,
    required=False,
    help="path to the model to resume training",
)
parser.add_argument(
    "--resume_run_id",
    type=str,
    default=None,
    required=False,
    help="wandb run id which is to be resumed",
)
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="offline",
    required=False,
    help="online, offline, disabled",
)

# Parse arguments
args = parser.parse_args()
config_path = args.config_path
data_src = args.data_src
gpu_index_to_use = args.gpu_index_to_use
train_techs = args.train_techs
resume_checkpoint_path = args.resume_checkpoint_path
resume_run_id = args.resume_run_id
n_rows = args.nrows
wandb_mode = args.wandb_mode

with open(config_path, "r") as file:
    config = json.load(file)
config["terminal_args"] = vars(args)

current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H-%M-%S")


model_output_dir = os.path.join(
    config.get("output").get("dir"),
    str(formatted_datetime),
)
os.makedirs(model_output_dir, exist_ok=True)
json.dump(config, open(os.path.join(model_output_dir, "config.json"), "w"), indent=4)
file = open(os.path.join(model_output_dir, config.get("output").get("log")), "w")

if gpu_index_to_use is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index_to_use

printd("*" * 10 + "GPUs" + "*" * 10, file=file)
for i in range(torch.cuda.device_count()):
    printd(torch.cuda.get_device_properties(i).name, file=file)
printd("*" * 30, file=file)

assert os.getenv("WANDB_LOG_MODEL") == "end"
wandb.login(key=os.getenv("WANDB_API_KEY"))
if resume_run_id is not None:
    wandb.init(
        project="mlm-fine-tuning",
        entity="nasa-impact",
        mode=wandb_mode,
        id=resume_run_id,
        resume="must",
    )
else:
    wandb.init(project="mlm-fine-tuning", entity="nasa-impact", mode=wandb_mode)

if __name__ == "__main__":
    model = get_model(config, train_techs, file)

    printd("*" * 10 + "Started DataPreprocessing" + "*" * 10, file=file)

    if any(config.get("input").get("dataset").get("kw_masking_type").values()):
        lm_dataset, tokenizer, data_collator = preprocess_dataset_with_kw_masking(
            config.get("input"),
            data_src,
            n_rows,
        )
    else:
        lm_dataset, tokenizer, data_collator = preprocess_dataset(
            config.get("input"),
            data_src,
            n_rows,
        )

    print(len(lm_dataset["train"]["input_ids"][0]))

    if resume_checkpoint_path is not None:
        output_dir = "/".join(resume_checkpoint_path.split("/")[:-1])
    else:
        output_dir = f"{config.get('output').get('model_backups_path')}timestamp_{formatted_datetime}/{config.get('input').get('model').get('hf')}/"
    training_args = TrainingArguments(
        output_dir=output_dir,
        **config.get("TrainingArguments"),
        label_names=["labels"],  # https://github.com/huggingface/peft/issues/1120,
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
    trainer.can_return_loss = True

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
    wandb.config.update(custom_cfg, allow_val_change=True)

    printd("*" * 10 + "Started Training" + "*" * 10, file=file)
    if resume_checkpoint_path is not None:
        trainer.train(resume_from_checkpoint=resume_checkpoint_path)
    else:
        trainer.train()

    model_save_loc = os.path.join(
        model_output_dir,
        config.get("output").get("final_model_path"),
    )
    os.makedirs(model_save_loc, exist_ok=True)
    trainer.model.save_pretrained(model_save_loc)
    tokenizer.save_pretrained(model_save_loc)

    printd("*" * 10 + "Started Evaluation" + "*" * 10, file=file)
    eval_report_df = generate_eval_table(trainer, lm_dataset)
    wandb.log({"eval_report": wandb.Table(dataframe=eval_report_df)})

    printd("*" * 10 + "Started Inference" + "*" * 10, file=file)
    inference_df = generate_inference(
        lm_dataset["test"],
        tokenizer,
        model_save_loc,
        top_k=config.get("output").get("inference").get("top_k"),
        n_predictions=config.get("output").get("inference").get("n_predictions"),
        max_length=config.get("input").get("dataset").get("chunk_size"),
        use_keyword_based_masking=any(
            config.get("input").get("dataset").get("kw_masking_type").values(),
        ),
    )
    wandb.log({"inference": wandb.Table(dataframe=inference_df)})
    wandb.finish()

file.close()
