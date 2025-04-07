import datetime
import os
import random
import sys
import tempfile

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

matplotlib.use("Agg")
from urllib.parse import urlparse

import matplotlib.pyplot as plt

from .reporting import (
    add_diff_hyperparameter,
    add_loss_comparision,
    add_metrics_comparison,
    create_overview_tab,
)


def set_seed(seed=42):
    """Set all seeds to make results reproducible (deterministic mode).
    When seed is a false-y value or not supplied, disables deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

from typing import Any, List, Optional, Union

from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    roc_auc_score,
    zero_one_loss,
)
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import wandb

current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H-%M-%S")
file = open("model_log.txt", "w")


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
        num_items_in_batch=None,
        return_outputs=False,
    ):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute focal loss
        loss = self.focal_binary_cross_entropy(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class CustomDataset(Dataset):
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


def printd(*args, **kwargs):
    if kwargs.get("file", None) is None:
        print(*args, **kwargs)
    else:
        print(*args, **kwargs)
        del kwargs["file"]
        print(*args, **kwargs)


printd("*" * 10 + "GPUs" + "*" * 10, file=file)
for i in range(torch.cuda.device_count()):
    printd(torch.cuda.get_device_properties(i).name, file=file)
printd("*" * 30, file=file)


def compute_metrics(pred, target_columns, threshold=0.2):
    labels = pred.label_ids
    logits = pred.predictions

    # Apply sigmoid to logits to get probabilities
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()

    # Apply threshold to get binary predictions
    predictions = (probabilities > threshold).astype(int)

    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)

    # Compute precision, recall, f1 for micro, macro, and weighted
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

    # Compute per-class precision, recall, f1
    report = classification_report(
        labels,
        predictions,
        output_dict=True,
        target_names=target_columns,
    )
    # print(report)

    return {
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "hamming_loss": hamming_loss(labels, predictions),
        "zero_one_loss": zero_one_loss(labels, predictions),
        "jaccard_score": jaccard_score(labels, predictions, average="weighted"),
    }


def get_prediction_table(
    model,
    tokenizer,
    concept_id,
    texts,
    target_columns,
    threshold=0.5,
    expected=None,
    batch_size=32,
):
    model.eval()
    num_batches = (len(texts) + batch_size - 1) // batch_size
    all_tables = []
    all_probablities = []

    for i in tqdm(range(num_batches)):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]
        batch_concept_id = concept_id[i * batch_size : (i + 1) * batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            try:
                logits = outputs.logits
            except:
                logits = outputs["logits"]

        probabilities = torch.sigmoid(logits.clone().detach()).cpu().numpy()
        batch_table = pd.DataFrame(dict(concept_id=batch_concept_id, text=batch_texts))

        probabilities_df = pd.DataFrame(probabilities, columns=target_columns)
        all_probablities.append(probabilities_df)

        all_tables.append(batch_table)
    return pd.concat(all_tables, ignore_index=True), pd.concat(
        all_probablities,
        ignore_index=True,
    )


def train_model(
    config,
    key,
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    train_input,
    val_input,
    test_input,
    train_target,
    val_target,
    test_target,
    data_collator,
    target_columns,
    tokenizer,
):
    resume = config.get(key).get("wandb").get("resume").get("resume")
    model_name = config.get("__common__").get("base_model_path")
    freeze_base = config.get(key).get("hyperparamaters").get("freeze_base")
    resume_checkpoint_path = (
        config.get(key).get("wandb").get("resume").get("last_checkpoint_path")
    )
    if resume_checkpoint_path is not None and resume:
        output_dir = "/".join(resume_checkpoint_path.split("/")[:-1])
    else:
        output_dir = f"tmp/models/timestamp_{formatted_datetime}/{model_name}_freeze_base_{freeze_base}/"

    training_args = TrainingArguments(
        per_device_train_batch_size=config.get(key)
        .get("hyperparamaters")
        .get("batch_size"),
        per_device_eval_batch_size=config.get(key)
        .get("hyperparamaters")
        .get("batch_size"),
        eval_strategy="steps",
        eval_steps=10,
        save_steps=10,
        logging_steps=10,
        output_dir=output_dir,
        num_train_epochs=config.get(key).get("hyperparamaters").get("epochs"),
        save_total_limit=2,
        save_strategy="steps",
        remove_unused_columns=True,
        logging_dir="./logs",
        optim="adamw_torch",
        learning_rate=config.get(key).get("hyperparamaters").get("learning_rate"),
        weight_decay=config.get(key).get("hyperparamaters").get("weight_decay"),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        report_to="wandb",
        load_best_model_at_end=True,  # Load the best model at the end of training based on evaluation metric
        metric_for_best_model="eval_loss",  # Metric to monitor for the best model
        greater_is_better=False,
        lr_scheduler_type=config.get(key)
        .get("hyperparamaters")
        .get("lr_scheduler_type"),  # for making model better
        warmup_ratio=config.get(key)
        .get("hyperparamaters")
        .get("warmup_ratio"),  # added this to improve the training,
        # gradient_accumulation_steps = 2, # added this to improve the training,
        fp16=False,
        bf16=True,
    )
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.get(key).get("hyperparamaters").get("patience"),
    )
    trainer_config = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, target_columns),
        callbacks=[early_stopping],
    )
    if config.get(key).get("hyperparamaters").get("loss_fn") == "focal_loss":
        trainer = CustomTrainer(
            **trainer_config,
            focal_loss_gamma=config.get(key)
            .get("hyperparamaters")
            .get("focal_loss_gamma"),
        )
    else:
        trainer = Trainer(
            **trainer_config,
        )
    printd("*" * 10 + "Starting training" + "*" * 10, file)
    if resume_checkpoint_path is not None and resume:
        printd("Resuming Traning from checkpoint", file=file)
        trainer.train(resume_from_checkpoint=resume_checkpoint_path)
    else:
        trainer.train()

    printd("*" * 10 + "Starting Traning evaluation" + "*" * 10, file=file)
    eval_report_map = dict(
        train=trainer.evaluate(train_dataset),
        val=trainer.evaluate(val_dataset),
        test=trainer.evaluate(test_dataset),
    )

    printd("*" * 10 + "Putting metrics to wandb" + "*" * 10, file=file)
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

    eval_datamap = dict(
        train=dict(
            concept_id=list(train_input.index),
            text=list(train_input.values.flatten()),
            expected=train_target,
        ),
        val=dict(
            concept_id=list(val_input.index),
            text=list(val_input.values.flatten()),
            expected=val_target,
        ),
        test=dict(
            concept_id=list(test_input.index),
            text=list(test_input.values.flatten()),
            expected=test_target,
        ),
    )

    printd("*" * 10 + "Putting metrics to wandb" + "*" * 10, file=file)
    for eval_type in eval_datamap:
        print(f">>>{eval_type}")
        table, probabilities_of = get_prediction_table(
            model=model,
            tokenizer=tokenizer,
            concept_id=eval_datamap[eval_type]["concept_id"],
            texts=eval_datamap[eval_type]["text"],
            target_columns=target_columns,
            expected=eval_datamap[eval_type]["expected"],
        )
        print(f"<<<{eval_type}")
        # table.to_parquet(f'./_df_{eval_type}.parquet')

        # Use a temporary file for compressed Parquet storage
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp_file:
            table.to_parquet(tmp_file.name)  # Save to temp file

            # Create artifact and log it
            artifact = wandb.Artifact(f"df_{eval_type}", type="dataset")
            artifact.add_file(tmp_file.name, name=f"df_{eval_type}.parquet")
            wandb.log_artifact(artifact)

    return trainer.model


def eval_model():
    pass


def custom_stratification_technique(
    data: pd.DataFrame,
    input_columns: List[str],
    target_columns: List[str],
    stratification_col: str,
    concept_id_column: str,
    val_size: float = 0.1,
    test_size: float = 0.1,
):
    assert stratification_col in data.columns

    def vstack_array(x, y):
        if x is None and y is None:
            return None
        elif x is None:
            return y
        elif y is None:
            return x
        else:
            return np.vstack((x, y))

    train_input, train_target, val_input, val_target, test_input, test_target = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    data = data.reset_index()
    for group_name, group_data in data.groupby(stratification_col):

        if group_data.shape[0] < 2:
            if random.choice(["train", "test"]) == "train":
                group_train_input = group_data[
                    [concept_id_column] + input_columns
                ].values
                group_train_target = group_data[target_columns].values
                group_test_input = None
                group_test_target = None
            else:
                group_train_input = None
                group_train_target = None
                group_test_input = group_data[
                    [concept_id_column] + input_columns
                ].values
                group_test_target = group_data[target_columns].values
        else:
            (
                group_train_input,
                group_train_target,
                group_test_input,
                group_test_target,
            ) = iterative_train_test_split(
                group_data[[concept_id_column] + input_columns].values,
                group_data[target_columns].values,
                test_size=(val_size + test_size),
            )

        if group_test_input is None:
            group_val_input = None
            group_val_target = None
        elif group_test_input.shape[0] < 2:
            if random.choice(["test", "val"]) == "val":
                group_val_input = group_test_input
                group_val_target = group_test_target
                group_test_input = None
                group_test_target = None
            else:
                group_val_input = None
                group_val_target = None
        else:
            (
                group_val_input,
                group_val_target,
                group_test_input,
                group_test_target,
            ) = iterative_train_test_split(
                group_test_input,
                group_test_target,
                test_size=(test_size / (val_size + test_size)),
            )

        train_input = vstack_array(train_input, group_train_input)
        train_target = vstack_array(train_target, group_train_target)
        val_input = vstack_array(val_input, group_val_input)
        val_target = vstack_array(val_target, group_val_target)
        test_input = vstack_array(test_input, group_test_input)
        test_target = vstack_array(test_target, group_test_target)

    return train_input, train_target, val_input, val_target, test_input, test_target


def get_stratified_splits(
    data: pd.DataFrame,
    input_columns: List[str],
    target_columns: List[str],
    stratification_col: str,
    concept_id_column: str,
    val_size: float = 0.1,
    test_size: float = 0.1,
):
    data = data.reset_index()
    if stratification_col == "" or stratification_col is None:
        train_input, train_target, test_input, test_target = iterative_train_test_split(
            data[[concept_id_column] + input_columns].values,
            data[target_columns].values,
            test_size=(val_size + test_size),
        )

        # again split for validation
        val_input, val_target, test_input, test_target = iterative_train_test_split(
            test_input,
            test_target,
            test_size=(test_size / (val_size + test_size)),
        )
    else:
        (
            train_input,
            train_target,
            val_input,
            val_target,
            test_input,
            test_target,
        ) = custom_stratification_technique(
            data,
            input_columns,
            target_columns,
            stratification_col,
            concept_id_column,
            val_size,
            test_size,
        )

    data.set_index(concept_id_column, inplace=True)

    return (
        pd.DataFrame(
            train_input,
            columns=[concept_id_column] + input_columns,
        ).set_index(concept_id_column),
        pd.DataFrame(train_target, columns=target_columns),
        pd.DataFrame(val_input, columns=[concept_id_column] + input_columns).set_index(
            concept_id_column,
        ),
        pd.DataFrame(val_target, columns=target_columns),
        pd.DataFrame(test_input, columns=[concept_id_column] + input_columns).set_index(
            concept_id_column,
        ),
        pd.DataFrame(test_target, columns=target_columns),
    )


def preprocess_data(config, key):
    printd("*" * 10 + "Started DataReading" + "*" * 10, file=file)
    data_path = config.get(key).get("data").get("path")
    data = pd.read_parquet(data_path).fillna(0)
    nrows = config.get("__common__").get("nrows")
    if nrows:
        data = data.head(nrows)
        # Remove columns where all values are 0
        data = data.loc[:, (data != 0).any(axis=0)]

    non_target_columns = ("concept-id", "provider-id", "Abstract")
    target_columns = [i for i in list(data.columns) if i not in non_target_columns]

    input_columns = ["Abstract"]
    concept_id_column = "concept-id"
    data.set_index(concept_id_column, inplace=True)
    num_labels = len(target_columns)
    stratification_col = (
        config.get(key).get("hyperparamaters").get("stratification_col")
    )

    printd("*" * 10 + "Started Stratification" + "*" * 10, file=file)
    (
        train_input,
        train_target,
        val_input,
        val_target,
        test_input,
        test_target,
    ) = get_stratified_splits(
        data,
        input_columns=input_columns,
        target_columns=target_columns,
        stratification_col=stratification_col,
        concept_id_column=concept_id_column,
    )
    printd("*" * 10 + "" + "*" * 10, file=file)

    printd(f"Train shapes: {train_input.shape}, {train_target.shape}", file=file)
    printd(f"Val shapes: {val_input.shape}, {val_target.shape}")
    printd(f"Test shapes: {test_input.shape}, {test_target.shape}")
    printd("*" * 10 + "" + "*" * 10, file=file)

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("__common__").get("base_model_path"),
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    input_to_encoding = lambda x: tokenizer(
        list(x.values.flatten()),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )

    train_encodings = input_to_encoding(train_input)
    val_encodings = input_to_encoding(val_input)
    test_encodings = input_to_encoding(test_input)

    train_dataset = CustomDataset(train_encodings, torch.tensor(train_target.values))
    val_dataset = CustomDataset(val_encodings, torch.tensor(val_target.values))
    test_dataset = CustomDataset(test_encodings, torch.tensor(test_target.values))
    assert len(train_dataset[0]["labels"]) == len(target_columns)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_input,
        val_input,
        test_input,
        train_target,
        val_target,
        test_target,
        data_collator,
        tokenizer,
        num_labels,
        target_columns,
        input_columns,
    )


def init_model(config, key, num_labels, target_columns):
    printd("*" * 10 + "Starting Model Init" + "*" * 10, file=file)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.get("__common__").get("base_model_path"),
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    # reconfig properly
    model.config.id2label = {idx: label for idx, label in enumerate(target_columns)}
    model.config.label2id = {label: idx for idx, label in enumerate(target_columns)}
    for param in model.base_model.parameters():
        param.requires_grad = (
            not config.get(key).get("hyperparamaters").get("freeze_base")
        )

    return model


def init_wandb(
    config,
    key,
    model,
    num_labels,
    target_columns,
    input_columns,
    train_input,
    val_input,
    test_input,
):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    resume = config.get(key).get("wandb").get("resume").get("resume")
    resume_run_id = config.get(key).get("wandb").get("resume").get("run_id")
    project_name = config.get(key).get("wandb").get("project")
    mode = config.get(key).get("wandb").get("mode")

    if resume_run_id is not None and resume:
        wandb.init(
            project=project_name,
            notes=config.get(key).get("desc"),
            tags=config.get(key).get("tags"),
            mode=mode,
            id=resume_run_id,
            resume="must",
        )
    else:
        wandb.init(
            project=project_name,
            mode=mode,
            notes=config.get(key).get("desc"),
            tags=config.get(key).get("tags"),
        )
    wandb.watch(model, log="all")

    # update the config to wandb
    custom_cfg = dict(
        model_name=config.get("__common__").get("base_model_path"),
        num_labels=num_labels,
        target_columns=target_columns,
        input_columns=input_columns,
        num_input_columns=len(input_columns),
        data_size=len(train_input) + len(val_input) + len(test_input),
        train_size=len(train_input),
        val_size=len(val_input),
        test_size=len(test_input),
    )
    custom_cfg = {**custom_cfg, **config.get(key).get("hyperparamaters")}
    wandb.config.update(custom_cfg, allow_val_change=True)


def train_and_eval_model(config, key="science_keyword_classification"):
    load_dotenv(config.get("__common__").get(".env_path"))

    # Reading Data
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_input,
        val_input,
        test_input,
        train_target,
        val_target,
        test_target,
        data_collator,
        tokenizer,
        num_labels,
        target_columns,
        input_columns,
    ) = preprocess_data(config, key)

    # Initializing Model
    model = init_model(config, key, num_labels, target_columns)

    # Initializing Wandb
    init_wandb(
        config,
        key,
        model,
        num_labels,
        target_columns,
        input_columns,
        train_input,
        val_input,
        test_input,
    )

    # Training Model
    model = train_model(
        config,
        key,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        train_input,
        val_input,
        test_input,
        train_target,
        val_target,
        test_target,
        data_collator,
        target_columns,
        tokenizer,
    )

    # finding the wandb run
    remove_domain = lambda url: urlparse(url).path
    new_wandb_run = {
        "name": config.get(key).get("name"),
        "desc": config.get(key).get("desc"),
        "run_id": remove_domain(wandb.run.url),
        "tokenizer_path": config.get("__common__").get("base_model_path"),
    }

    wandb.finish()

    # start to work on reporting only if the current run is logged to wandb online
    if config.get(key).get("wandb").get("mode") == "online":
        previous_wandb_runs = config.get(key).get("previous_model").get("wandb_runs")

        wb = create_overview_tab(new_wandb_run, previous_wandb_runs)
        wb = add_diff_hyperparameter(wb, new_wandb_run, previous_wandb_runs)
        wb = add_loss_comparision(wb, new_wandb_run, previous_wandb_runs)
        wb = add_metrics_comparison(
            wb,
            new_wandb_run,
            previous_wandb_runs,
            train_df=pd.concat(
                [train_input.reset_index(), train_target.reset_index(drop=True)],
                axis=1,
            ),
            test_df=pd.concat(
                [test_input.reset_index(), test_target.reset_index(drop=True)],
                axis=1,
            ),
            text_col=input_columns[0],
            label_cols=target_columns,
            top_ks=config.get(key).get("eval_config").get("topks"),
            thresholds=config.get(key).get("eval_config").get("thresholds"),
            key=key,
        )

        report_dir = config.get("__common__").get("report_dir")
        os.makedirs(report_dir, exist_ok=True)
        wb.save(f"{report_dir}/{key}_report.xlsx")

    else:
        # If wandb mode is not online, we don't need to do any reporting
        print("Wandb mode is not online, skipping reporting.")

    return model, None, [new_wandb_run]
