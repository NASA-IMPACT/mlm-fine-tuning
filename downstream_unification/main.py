import argparse
import json
import sys

import yaml
from trainers.train import HFTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="config.json", required=False)
parser.add_argument("--downstreams", type=str, nargs="*", default=None, required=False)
parser.add_argument(
    "--wandb_mode",
    type=str,
    default=None,
    required=False,
    help="Overrides for all downstream if given: online, offline, disabled",
)
parser.add_argument("--nrows", type=int, default=100, required=False)

args = parser.parse_args()
config_path = args.config_path
downstreams = args.downstreams
wandb_mode = args.wandb_mode
nrows = args.nrows

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# manage the config values
downstreams = (
    [i for i in config.keys() if i != "__common__"] if not downstreams else downstreams
)
config["__common__"]["nrows"] = nrows

if wandb_mode:
    for downstream in downstreams:
        config[downstream]["wandb"]["mode"] = wandb_mode


# if __name__ == "__main__":
#     for downstream in downstreams:
#         if downstream == "science_keyword_classification":
#             model, evals, wandb_runs = train_and_eval_skc(config)
#         else:
#             print(f"Downstream '{downstream}' not found in downstreams")

if __name__ == "__main__":
    for downstream in downstreams:
        trainer = HFTrainer(config, downstream)
        trainer.train()
