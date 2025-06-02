import datetime
import importlib
import os
import random
import tempfile
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from reporting import (
    add_diff_hyperparameter,
    add_loss_comparision,
    add_metrics_comparison,
    create_overview_tab,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    roc_auc_score,
    zero_one_loss,
)
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import wandb


def set_seed(seed=42):
    """Set all seeds to make results reproducible (deterministic mode).
    When seed is a false-y value or not supplied, disables deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class FocalLossTrainer(Trainer):
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


class BaseTrainer(ABC):
    def __init__(self, config, key):
        self.config = config
        self.key = key
        self.current_datetime = datetime.datetime.now()
        set_seed(self.config.get(self.key).get("seed"))
        load_dotenv(self.config.get("__common__").get(".env_path"))

        self.target_columns = None  # needs to be defined in subclasses

    def generate_report(self):
        # finding the wandb run
        remove_domain = lambda url: urlparse(url).path
        new_wandb_run = {
            "name": self.config.get(self.key).get("name"),
            "desc": self.config.get(self.key).get("desc"),
            "run_id": remove_domain(wandb.run.url),
            "tokenizer_path": self.config.get("__common__").get("base_model_path"),
        }
        wandb.finish()

        # start to work on reporting only if the current run is logged to wandb online
        if self.config.get(self.key).get("wandb").get("mode") == "online":
            previous_wandb_runs = (
                self.config.get(self.key).get("previous_model").get("wandb_runs")
            )

            wb = create_overview_tab(new_wandb_run, previous_wandb_runs)
            wb = add_diff_hyperparameter(wb, new_wandb_run, previous_wandb_runs)
            wb = add_loss_comparision(wb, new_wandb_run, previous_wandb_runs)

            wb = add_metrics_comparison(
                wb,
                new_wandb_run,
                previous_wandb_runs,
                train_df=pd.concat(
                    [
                        self.data.get("train").get("x").reset_index(),
                        self.data.get("train").get("y").reset_index(drop=True),
                    ],
                    axis=1,
                ),
                test_df=pd.concat(
                    [
                        self.data.get("test").get("x").reset_index(),
                        self.data.get("test").get("y").reset_index(drop=True),
                    ],
                    axis=1,
                ),
                text_col=self.config.get(self.key).get("data").get("input_text_col"),
                label_cols=self.target_columns,
                top_ks=self.config.get(self.key).get("eval_config").get("topks"),
                thresholds=self.config.get(self.key)
                .get("eval_config")
                .get("thresholds"),
                key=self.key,
            )

            report_dir = self.config.get("__common__").get("report_dir")
            os.makedirs(report_dir, exist_ok=True)
            wb.save(f"{report_dir}/{self.key}_report.xlsx")

        else:
            # If wandb mode is not online, we don't need to do any reporting
            print("Wandb mode is not online, skipping reporting.")


class HFTrainer(BaseTrainer):
    def __init__(self, config, key):
        super().__init__(config, key)

    def column_stratification_for_multilabel(
        self,
        data: pd.DataFrame,
        input_columns: List[str],
        target_columns: List[str],
        stratification_col: str,
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
                    group_train_input = group_data[input_columns].values
                    group_train_target = group_data[target_columns].values
                    group_test_input = None
                    group_test_target = None
                else:
                    group_train_input = None
                    group_train_target = None
                    group_test_input = group_data[input_columns].values
                    group_test_target = group_data[target_columns].values
            else:
                (
                    group_train_input,
                    group_train_target,
                    group_test_input,
                    group_test_target,
                ) = iterative_train_test_split(
                    group_data[input_columns].values,
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

    def get_splits_for_multilabel(self, data: pd.DataFrame):
        stratification_col = (
            self.config.get(self.key).get("data").get("stratification_col")
        )
        self.input_columns = [
            self.config.get(self.key).get("data").get("input_text_col"),
        ]
        self.non_target_columns = (
            self.config.get(self.key).get("data").get("non_target_cols")
        )
        self.target_columns = [
            c for c in data.columns if c not in self.non_target_columns
        ]
        test_size = self.config.get(self.key).get("data").get("test_size")
        val_size = self.config.get(self.key).get("data").get("val_size", 0)

        data = data.reset_index()
        if stratification_col == "" or stratification_col is None:
            (
                train_input,
                train_target,
                test_input,
                test_target,
            ) = iterative_train_test_split(
                data[self.input_columns].values,  # X
                data[self.target_columns].values,  # y
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
            ) = self.column_stratification_for_multilabel(
                data,
                self.input_columns,
                self.target_columns,
                stratification_col,
                val_size,
                test_size,
            )

        return (
            pd.DataFrame(train_input, columns=self.input_columns),
            pd.DataFrame(train_target, columns=self.target_columns),
            pd.DataFrame(val_input, columns=self.input_columns),
            pd.DataFrame(val_target, columns=self.target_columns),
            pd.DataFrame(test_input, columns=self.input_columns),
            pd.DataFrame(test_target, columns=self.target_columns),
            {
                "input_columns": self.input_columns,
                "target_columns": self.target_columns,
                "non_target_columns": self.non_target_columns,
            },
        )

    def get_splits_for_multiclass(self, data: pd.DataFrame):
        stratification_col = (
            self.config.get(self.key).get("data").get("stratification_col")
        )
        input_columns = [self.config.get(self.key).get("data").get("input_text_col")]
        non_target_columns = (
            self.config.get(self.key).get("data").get("non_target_cols")
        )
        target_columns = [c for c in data.columns if c not in non_target_columns]
        test_size = self.config.get(self.key).get("data").get("test_size")
        val_size = self.config.get(self.key).get("data").get("val_size", 0)

        data = data.reset_index()

        # add a temp label column to the data
        data["_temp_label_"] = data[target_columns].values.argmax(axis=1)

        if stratification_col == "" or stratification_col is None:
            train_df, test_df = train_test_split(
                data,
                test_size=test_size + val_size,
                stratify=data["_temp_label_"],
                random_state=self.config.get(self.key).get("seed"),
            )
            val_df, test_df = train_test_split(
                test_df,
                test_size=test_size / (val_size + test_size),
                stratify=test_df["_temp_label_"],
                random_state=self.config.get(self.key).get("seed"),
            )
            train_x = train_df[input_columns]
            train_y = train_df[target_columns]
            val_x = val_df[input_columns]
            val_y = val_df[target_columns]
            test_x = test_df[input_columns]
            test_y = test_df[target_columns]
        else:
            raise NotImplementedError(
                "Stratification is not implemented for multiclass / binary classification. Please try without any stratification column.",
            )
        return (
            train_x,
            train_y,
            val_x,
            val_y,
            test_x,
            test_y,
            {
                "input_columns": input_columns,
                "target_columns": target_columns,
                "non_target_columns": non_target_columns,
            },
        )

    def split_data(self, df: pd.DataFrame):
        # check if the problem type is multilabel
        if (
            self.config.get(self.key).get("problem_type")
            == "multi_label_classification"
        ):
            (
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                test_y,
                var_info,
            ) = self.get_splits_for_multilabel(df)
        # check if the probelm type is multiclass / binary
        elif self.config.get(self.key).get("problem_type") in [
            "binary_classification",
            "multi_class_classification",
        ]:
            (
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                test_y,
                var_info,
            ) = self.get_splits_for_multiclass(df)
        else:
            raise NotImplementedError(
                f"Problem type '{self.config.get(self.key).get('problem_type')}' is not Implemented. Please use 'multi_label_classification' or 'binary_classification' or 'multi_class_classification'.",
            )

        if val_x.shape[0] == 0:
            print(
                "*" * 5
                + "Validation set is empty. Using test set as validation set."
                + "*" * 5,
            )
            val_x = test_x
            val_y = test_y
        return train_x, train_y, val_x, val_y, test_x, test_y, var_info

    def get_data(self):
        path = self.config.get(self.key).get("data").get("path")
        fmt = self.config.get(self.key).get("data").get("format").lower()

        if fmt == "csv":
            df = pd.read_csv(path)
        elif fmt == "parquet":
            df = pd.read_parquet(path)
        elif fmt == "json":
            df = pd.read_json(path)
        else:
            raise ValueError(
                f"Format '{fmt}' is not supported. Please use 'csv', 'parquet', or 'json'.",
            )
        return df

    def preprocess_data(self):
        print("*" * 10 + "Preprocessing" + "*" * 10)
        df = self.get_data()
        nrows = self.config.get("__common__").get("nrows")
        # check if custom data preprocessor is provided
        if self.config.get(self.key).get("data").get("custom_preprocessor"):
            module_name = ".".join(
                self.config.get(self.key)
                .get("data")
                .get("custom_preprocessor")
                .split(".")[:-1],
            )
            method_name = (
                self.config.get(self.key)
                .get("data")
                .get("custom_preprocessor")
                .split(".")[-1]
            )
            module = importlib.import_module(module_name)
            method = getattr(module, method_name)
            train_x, train_y, val_x, val_y, test_x, test_y, var_info = method(
                df,
                self.config.get(self.key).get("data"),
                seed=self.config.get(self.key).get("seed"),
                nrows=nrows,
            )
        else:
            if nrows:
                df = df.head(nrows)
            # split data for train, test and val
            train_x, train_y, val_x, val_y, test_x, test_y, var_info = self.split_data(
                df,
            )

        # Dynamically set attributes for every key in info_dict:
        for key, value in var_info.items():
            setattr(self, key, value)

        print(f"Train shapes: {train_x.shape}, {train_y.shape}")
        print(f"Val shapes: {val_x.shape}, {val_y.shape}")
        print(f"Test shapes: {test_x.shape}, {test_y.shape}")
        print("*" * 10 + "" + "*" * 10)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("__common__").get("base_model_path"),
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        input_to_encoding = lambda x: tokenizer(
            list(x.values.flatten()),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get(self.key)
            .get("additional_hyperparamaters")
            .get("tokenizer_max_length"),
            return_attention_mask=True,
        )

        train_encodings = input_to_encoding(train_x)
        val_encodings = input_to_encoding(val_x)
        test_encodings = input_to_encoding(test_x)

        train_dataset = CustomDataset(train_encodings, torch.tensor(train_y.values))
        val_dataset = CustomDataset(val_encodings, torch.tensor(val_y.values))
        test_dataset = CustomDataset(test_encodings, torch.tensor(test_y.values))

        data = {
            "train": {"x": train_x, "y": train_y, "tokenized": train_dataset},
            "val": {"x": val_x, "y": val_y, "tokenized": val_dataset},
            "test": {"x": test_x, "y": test_y, "tokenized": test_dataset},
        }

        return data, tokenizer, data_collator

    def init_model(self, num_labels: int):
        if (
            self.config.get(self.key).get("problem_type")
            == "multi_label_classification"
        ):
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.get("__common__").get("base_model_path"),
                num_labels=num_labels,
                problem_type="multi_label_classification",
            )

        elif self.config.get(self.key).get("problem_type") in [
            "binary_classification",
            "multi_class_classification",
        ]:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.get("__common__").get("base_model_path"),
                num_labels=num_labels,
                problem_type="single_label_classification",  # This is the correct setting for both binary & multi-class
            )

        else:
            raise NotImplementedError(
                f"Problem type '{self.config.get(self.key).get('problem_type')}' is not Implemented. Please use 'multi_label_classification' or 'binary_classification' or 'multi_class_classification'.",
            )

        # reconfig properly
        target_columns = self.data.get("train").get("y").columns.tolist()
        model.config.id2label = {idx: label for idx, label in enumerate(target_columns)}
        model.config.label2id = {label: idx for idx, label in enumerate(target_columns)}
        for param in model.base_model.parameters():
            param.requires_grad = (
                not self.config.get(self.key)
                .get("additional_hyperparamaters")
                .get("freeze_base", False)
            )

        return model

    def init_wandb(self):
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        resume = self.config.get(self.key).get("wandb").get("resume").get("resume")
        resume_run_id = (
            self.config.get(self.key).get("wandb").get("resume").get("run_id")
        )
        project_name = self.config.get(self.key).get("wandb").get("project")
        mode = self.config.get(self.key).get("wandb").get("mode")

        if resume_run_id is not None and resume:
            wandb.init(
                project=project_name,
                notes=self.config.get(self.key).get("desc"),
                tags=self.config.get(self.key).get("tags"),
                mode=mode,
                id=resume_run_id,
                resume="must",
            )
        else:
            wandb.init(
                project=project_name,
                mode=mode,
                notes=self.config.get(self.key).get("desc"),
                tags=self.config.get(self.key).get("tags"),
            )
        wandb.watch(self.model, log="all")

        n_train = self.data.get("train").get("x").shape[0]
        n_val = self.data.get("val").get("x").shape[0]
        n_test = self.data.get("test").get("x").shape[0]

        # update the config to wandb
        custom_cfg = dict(
            model_name=self.config.get("__common__").get("base_model_path"),
            num_labels=self.data.get("train").get("y").shape[1],
            target_columns=self.data.get("train").get("y").columns.tolist(),
            input_columns=self.data.get("train").get("x").columns.tolist(),
            num_input_columns=len(self.data.get("train").get("x").columns.tolist()),
            data_size=n_train + n_val + n_test,
            train_size=n_train,
            val_size=n_val,
            test_size=n_test,
            config_dump=self.config.get(self.key),
        )
        custom_cfg = {**custom_cfg}
        wandb.config.update(custom_cfg, allow_val_change=True)

    @staticmethod
    def compute_metrics_for_multilabel_multiclass(pred, target_columns, threshold=0.2):
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

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "precision_micro": precision_micro,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "recall_macro": recall_macro,
            "recall_micro": recall_micro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "classification_report": report,
            "hamming_loss": hamming_loss(labels, predictions),
            "zero_one_loss": zero_one_loss(labels, predictions),
            "jaccard_score": jaccard_score(labels, predictions, average="weighted"),
        }

    def compute_metrics(self, pred, threshold=0.2):
        target_columns = self.data.get("train").get("y").columns.tolist()
        # check if the problem type is multilabel
        if self.config.get(self.key).get("problem_type") in [
            "binary_classification",
            "multi_class_classification",
            "multi_label_classification",
        ]:
            return HFTrainer.compute_metrics_for_multilabel_multiclass(
                pred,
                target_columns,
                threshold,
            )

        raise NotImplementedError(
            f"Problem type '{self.config.get(self.key).get('problem_type')}' is not Implemented. Please use 'multi_label_classification' or 'binary_classification' or 'multi_class_classification'.",
        )

    def get_prediction_table(self, data_split, batch_size=32):
        self.model.eval()
        texts = list(self.data.get(data_split).get("x").values.flatten())
        num_batches = (len(texts) + batch_size - 1) // batch_size
        target_columns = self.data.get(data_split).get("y").columns.tolist()
        all_tables = []
        all_probablities = []

        for i in tqdm(range(num_batches)):
            self.data.get(data_split).get("x").reset_index(drop=True)
            batch_texts = texts[i * batch_size : (i + 1) * batch_size]
            # batch_concept_id = concept_id[i * batch_size : (i + 1) * batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.config.get(self.key)
                .get("additional_hyperparamaters")
                .get("tokenizer_max_length"),
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                try:
                    logits = outputs.logits
                except:
                    logits = outputs["logits"]

            probabilities = torch.sigmoid(logits.clone().detach()).cpu().numpy()
            batch_table = pd.DataFrame(
                dict(text=batch_texts),
            )

            probabilities_df = pd.DataFrame(probabilities, columns=target_columns)
            all_probablities.append(probabilities_df)

            all_tables.append(batch_table)
        return pd.concat(all_tables, ignore_index=True), pd.concat(
            all_probablities,
            ignore_index=True,
        )

    def train_model(self):
        resume = self.config.get(self.key).get("wandb").get("resume").get("resume")
        model_name = self.config.get("__common__").get("base_model_path")
        freeze_base = (
            self.config.get(self.key)
            .get("additional_hyperparamaters")
            .get("freeze_base", False)
        )
        formatted_datetime = self.current_datetime.strftime("%Y%m%d_%H-%M-%S")

        resume_checkpoint_path = (
            self.config.get(self.key)
            .get("wandb")
            .get("resume")
            .get("last_checkpoint_path")
        )
        if resume_checkpoint_path is not None and resume:
            output_dir = "/".join(resume_checkpoint_path.split("/")[:-1])
        else:
            output_dir = f"tmp/models/timestamp_{formatted_datetime}/{model_name}_freeze_base_{freeze_base}/"

        training_args = TrainingArguments(
            output_dir=output_dir,
            **self.config.get(self.key).get("hyperparamaters"),
        )
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config.get(self.key)
            .get("additional_hyperparamaters")
            .get("patience"),
        )
        trainer_config = dict(
            model=self.model,
            args=training_args,
            train_dataset=self.data.get("train").get("tokenized"),
            eval_dataset=self.data.get("val").get("tokenized"),
            data_collator=self.data_collator,
            compute_metrics=lambda x: self.compute_metrics(x),
            callbacks=[early_stopping],
        )
        # check loss function
        loss_fn = (
            self.config.get(self.key)
            .get("additional_hyperparamaters")
            .get("loss_fn", "bce")
        )
        if loss_fn == "focal_loss":
            trainer = FocalLossTrainer(
                **trainer_config,
                focal_loss_gamma=self.config.get(self.key)
                .get("additional_hyperparamaters")
                .get("focal_loss_gamma"),
            )
        elif loss_fn == "bce":
            trainer = Trainer(
                **trainer_config,
            )
        else:
            raise NotImplementedError(
                f"Loss function '{loss_fn}' is not Implemented. Please use 'focal_loss' or 'bce'.",
            )
        print("*" * 10 + "Starting training" + "*" * 10)
        if resume_checkpoint_path is not None and resume:
            print("Resuming Traning from checkpoint")
            trainer.train(resume_from_checkpoint=resume_checkpoint_path)
        else:
            trainer.train()

        print("*" * 10 + "Starting evaluation" + "*" * 10)
        eval_report_map = dict(
            train=trainer.evaluate(self.data.get("train").get("tokenized")),
            val=trainer.evaluate(self.data.get("val").get("tokenized")),
            test=trainer.evaluate(self.data.get("test").get("tokenized")),
        )

        print("*" * 10 + "Putting metrics to wandb" + "*" * 10)
        eval_dct = {}
        for eval_type, eval_report in eval_report_map.copy().items():
            classification_report_df = pd.DataFrame(
                eval_report.get("eval_classification_report"),
            )
            print(f">>> {eval_type}")
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

        # making some inference
        for split_type in self.data.keys():
            print(f">>>{split_type}")
            table, probabilities_of = self.get_prediction_table(split_type)
            print(f"<<<{split_type}")

            # Use a temporary file for compressed Parquet storage
            with tempfile.NamedTemporaryFile(
                suffix=".parquet",
                delete=True,
            ) as tmp_file:
                table.to_parquet(tmp_file.name)  # Save to temp file

                # Create artifact and log it
                artifact = wandb.Artifact(f"df_{eval_type}", type="dataset")
                artifact.add_file(tmp_file.name, name=f"df_{eval_type}.parquet")
                wandb.log_artifact(artifact)

            # also add probabilities
            with tempfile.NamedTemporaryFile(
                suffix=".parquet",
                delete=True,
            ) as tmp_file:
                probabilities_of.to_parquet(tmp_file.name)  # Save to temp file

                # Create artifact and log it
                artifact = wandb.Artifact(f"df_prob_{eval_type}", type="dataset")
                artifact.add_file(tmp_file.name, name=f"df_prob_{eval_type}.parquet")
                wandb.log_artifact(artifact)

        return trainer.model

    def train(self):
        self.data, self.tokenizer, self.data_collator = self.preprocess_data()
        self.model = self.init_model(
            num_labels=self.data.get("train").get("y").shape[1],
        )
        self.init_wandb()
        _ = self.train_model()
        self.generate_report()
