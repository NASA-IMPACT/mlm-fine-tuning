import math

import pandas as pd


def generate_eval_table(trainer, dataset):
    eval_report = {}
    for data_split, ds in dataset.items():
        eval_report["data_split"] = data_split
        eval_report["loss"] = trainer.evaluate(eval_dataset=ds)["eval_loss"]
        eval_report["perplexity"] = math.exp(eval_report["loss"])
    return pd.DataFrame(eval_report)
