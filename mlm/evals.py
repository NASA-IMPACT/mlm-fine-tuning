import math
from typing import Dict

import pandas as pd
from datasets import DatasetDict
from transformers import Trainer


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
        eval_report["loss"].append(
            trainer.evaluate(eval_dataset=ds)["eval_loss"],
        )
        eval_report["perplexity"].append(math.exp(eval_report["loss"][-1]))

    return pd.DataFrame(eval_report)
