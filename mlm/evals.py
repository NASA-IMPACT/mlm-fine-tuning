import math
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from transformers import EvalPrediction, Trainer


def generate_eval_table(trainer: Trainer, lm_dataset: DatasetDict) -> pd.DataFrame:
    """
    Generates an evaluation table with loss and perplexity for each data split.

    Args:
        trainer (Trainer): A Hugging Face Trainer instance used for evaluation.
        lm_dataset (DatasetDict): A dataset dictionary containing splits (train, validation, test).

    Returns:
        pd.DataFrame: A DataFrame with evaluation metrics for each data split.
    """
    eval_report: Dict[str, list] = {
        "data_split": [],
        "loss": [],
        "perplexity": [],
    }

    for data_split, ds in lm_dataset.items():
        eval_report["data_split"].append(data_split)
        t_ = trainer.evaluate(eval_dataset=ds)
        print(t_, data_split)
        eval_report["loss"].append(
            t_["eval_loss"],
        )
        eval_report["perplexity"].append(math.exp(eval_report["loss"][-1]))

    return pd.DataFrame(eval_report)


def compute_metrics(p: EvalPrediction):
    """
    Computes the metrics (in this case, loss) for evaluation.

    Args:
        p: A tuple containing predictions and labels.

    Returns:
        A dictionary with the computed metrics.
    """
    predictions, labels = p

    print(predictions.shape, labels.shape)

    # Convert numpy arrays to tensors (if they aren't already)
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels).long()  # Ensure labels are long (int) type

    # Flatten the predictions and labels
    predictions = predictions.view(
        -1,
        predictions.size(-1),
    )  # Shape: [batch_size * seq_length, vocab_size]
    labels = labels.view(-1)  # Shape: [batch_size * seq_length]

    # CrossEntropyLoss expects logits and class indices
    loss_fct = torch.nn.CrossEntropyLoss()

    # Compute the loss
    loss = loss_fct(predictions, labels)
    return {"loss": loss.item()}
