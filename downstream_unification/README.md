# Multi-Task Fine-Tuning and Evaluation Framework

This project provides a configurable framework for fine-tuning and evaluating Hugging Face Transformer models on various downstream classification tasks. It includes robust data preprocessing, training, comprehensive evaluation, and automated report generation with Weights & Biases (W&B) integration.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Configuration (`config.yaml`)](#configuration-configyaml)
  - [Common Configuration (`__common__`)](#common-configuration-__common__)
  - [Downstream Task Configuration](#downstream-task-configuration)
- [Running the Code](#running-the-code)
  - [Training](#training)
- [Key Components](#key-components)
  - [`main.py`](#mainpy)
  - [`trainers/train.py`](#trainerstrainpy)
  - [`evaluators/multilabel_classifier_evaluation.py`](#evaluatorsmultilabel_classifier_evaluationpy)
  - [`reporting.py`](#reportingpy)
  - [`custom_preprocessors/preprocess.py`](#custom_preprocessorspreprocesspy)
- [Dependencies](#dependencies)
- [Contributing (Optional)](#contributing-optional)
- [License (Optional)](#license-optional)

---

## Project Overview

This framework is designed to streamline the process of adapting pre-trained transformer models to specific classification tasks. It supports:
-   **Multi-label classification**
-   **Multi-class classification**
-   **Binary classification**

Key functionalities include dynamic configuration loading, flexible data preprocessing (including custom scripts), model training with Hugging Face `Trainer`, detailed evaluation using a variety of metrics, and comparative report generation in Excel format, all integrated with Weights & Biases for experiment tracking and versioning.

---

## Features

✨ **Configurable Pipelines:** Easily define and manage multiple downstream tasks and their configurations through a central `config.yaml` file.
📊 **Comprehensive Evaluation:** Utilizes a wide array of metrics including precision, recall, F1-score (micro, macro, weighted), Jaccard similarity, Hamming loss, and Haystack-based document retrieval metrics (Recall, MRR, NDCG, MAP).
🔄 **Data Preprocessing:** Supports standard data formats (CSV, Parquet, JSON) and allows for custom preprocessing logic per downstream task.
🚀 **Hugging Face Integration:** Leverages `transformers` library for models, tokenizers, and the `Trainer` API.
📉 **Custom Loss Functions:** Includes an implementation for Focal Loss alongside standard Binary Cross-Entropy.
🔍 **Weights & Biases Logging:** Seamless integration with W&B for logging metrics, hyperparameters, model checkpoints, and evaluation artifacts.
📝 **Automated Reporting:** Generates detailed Excel reports comparing different W&B runs, including hyperparameter differences, loss curves, and evaluation metric comparisons with plots.
⏱️ **Evaluation Caching:** Caches evaluation results to speed up re-runs and avoid redundant computations.
🌱 **Reproducibility:** Supports setting global random seeds for reproducible results.
 strat **Advanced Data Splitting:** Implements iterative stratification for multi-label datasets and standard stratification for multi-class tasks.

---

## Directory Structure

```text
.
├── .env                      # Environment variables (API keys, etc.)
├── config.yaml               # Configuration file for all downstream tasks and common settings
├── custom_preprocessors/
│   └── preprocess.py         # Custom data preprocessing functions
├── evaluators/
│   └── multilabel_classifier_evaluation.py # Evaluation logic for classification tasks
├── main.py                   # Main script to run training pipelines
├── reporting.py              # Script for generating comparison reports
├── requirements.txt          # Python dependencies
└── trainers/
    └── train.py              # Core training logic using Hugging Face Trainer
```

---

## Setup

### Prerequisites

-   Python 3.8+
-   Access to a machine with necessary compute resources (GPU recommended for faster training).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NASA-IMPACT/mlm-fine-tuning
    git checkout develop
    cd mlm-fine-tuning/downstream_unification
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Environment Variables

Create a `.env` file in the root directory by copying the example:
```bash
export WANDB_LOG_MODEL="end"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export HUGGINGFACE_TOKEN="YOUR_HUGGINGFACE_HUB_TOKEN" # Optional, if using private models
```

The script will load these variables automatically. The path to the `.env` file can be configured in `config.yaml` under `__common__: .env_path`.

---

## Configuration (`config.yaml`)

The `config.yaml` file is the central place to define all parameters for your training runs. It's divided into a `__common__` section for shared settings and individual sections for each downstream task.

### Common Configuration (`__common__`)

This section contains parameters that are shared across all downstream tasks.

```yaml
__common__:
  base_model_path: nasa-impact/nasa-smd-ibm-v0.1  # Default base model for all tasks
  nrows: null  # Limit number of rows for training/evaluation (e.g., 100 for debugging, null for all data)
  .env_path: /path/to/your/.env  # Path to your .env file
  report_dir: ./reports/  # Directory to save generated Excel reports
```

### Downstream Task Configuration

Each downstream task has its own section in `config.yaml` (e.g., `science_keyword_classification`, `tdamm`).

```yaml
science_keyword_classification: # Unique name for the downstream task
  name: new_model_skc             # Name of the current W&B run/model version for this task
  desc: Finetuned model for science keyword classification # Description for W&B
  tags: [classification, science, keywords] # Tags for W&B
  problem_type: multi_label_classification # Type of problem: multi_label_classification, multi_class_classification, binary_classification
  seed: 42                        # Random seed for reproducibility

  data:
    path: /path/to/data.parquet   # Path to the training data file
    format: parquet                 # Data format: parquet, csv, json
    custom_preprocessor: null       # Optional: Python path to a custom preprocessing function (e.g., custom_preprocessors.preprocess.my_func)
    input_text_col: Abstract        # Column name for input text
    test_size: 0.1                  # Proportion of data for the test set
    val_size: 0.1                   # Proportion of data for the validation set (if null, val is same as test)
    stratification_col: provider-id # Column to use for stratified splitting (null if no stratification)
    non_target_cols: [concept-id, provider-id, Abstract] # List of columns that are NOT target labels

  wandb:
    project: science-keywords-recommender # W&B project name
    mode: online                    # W&B mode: online, offline, disabled
    resume:
      resume: false                 # Whether to resume a W&B run
      run_id: null                  # W&B run ID to resume (if resume is true)
      last_checkpoint_path: null    # Path to a local checkpoint to resume training from

  hyperparameters: # Arguments passed directly to Hugging Face TrainingArguments
    per_device_train_batch_size: 4
    per_device_eval_batch_size: 4
    eval_strategy: steps
    eval_steps: 10
    save_steps: 10
    logging_steps: 10
    num_train_epochs: 10
    save_total_limit: 2
    save_strategy: steps
    remove_unused_columns: true
    logging_dir: ./logs
    optim: adamw_torch
    learning_rate: 2.0e-5
    weight_decay: 0.1
    overwrite_output_dir: true
    do_train: true
    do_eval: true
    report_to: wandb
    load_best_model_at_end: true
    metric_for_best_model: eval_loss
    greater_is_better: false
    lr_scheduler_type: cosine
    warmup_ratio: 0.06
    fp16: false                     # Set to true for mixed-precision training (if GPU supports)
    bf16: true                      # Set to true for bfloat16 training (if GPU supports)
    gradient_accumulation_steps: 1

  additional_hyperparameters: # Custom hyperparameters used by the HFTrainer
    freeze_base: false              # Whether to freeze the base model layers
    patience: 4                     # Early stopping patience
    loss_fn: focal_loss             # Loss function: focal_loss or bce (Binary Cross-Entropy)
    focal_loss_gamma: 2.0           # Gamma parameter for focal loss (if used)
    tokenizer_max_length: 512       # Max sequence length for tokenizer

  eval_config: # Configuration for evaluation reporting
    topks: [1, 3, 5]                # Top-k values for which to compute metrics
    thresholds: [0.5, 0.6, 0.7]     # Thresholds for classification metrics

  previous_model: # Information about previous W&B runs for comparison in reports
    wandb_runs:
      - name: alpha-1.2.1
        desc: Best model fine-tuned previously
        run_id: /your_entity/your_project/run_id_example
        tokenizer_path: nasa-impact/nasa-smd-ibm-v0.1 # Tokenizer used for this previous model
```

## Running the Code

### Training

The main script `main.py` is used to initiate training runs.

```bash
python main.py [OPTIONS]
```

**Command-line Arguments:**

* `--config_path` (str, optional): Path to the configuration YAML file. Defaults to `config.yaml`.
* `--downstreams` (str, nargs="\*", optional): A list of downstream task keys (defined in `config.yaml`) to run. If not provided, all tasks in the config file (excluding `__common__`) will be run.
    * Example: `python main.py --downstreams science_keyword_classification tdamm`
* `--wandb_mode` (str, optional): Overrides the `wandb.mode` for all specified downstream tasks. Options: `online`, `offline`, `disabled`.
    * Example: `python main.py --wandb_mode offline`
* `--nrows` (int, optional): Overrides the `__common__.nrows` setting, limiting the number of data rows used for each task. Useful for quick debugging. Defaults to value in `config.yaml` or `100` if not set there.
    * Example: `python main.py --nrows 500`

**Example Usage:**

1. Run all downstream tasks defined in `config.yaml`:

    ```bash
    python main.py
    ```
2. Run specific downstream tasks:

    ```bash
    python main.py --downstreams science_keyword_classification
    ```
3. Run a specific task with W&B disabled and using only 1000 rows:
    ```bash
    python main.py --downstreams tdamm --wandb_mode disabled --nrows 1000
    ````

Upon completion of a training run (if `wandb.mode` is `online`), an Excel report comparing the current run with specified `previous_model.wandb_runs` will be generated in the directory specified by `__common__.report_dir`.

---

## Key Components

### `main.py`

* **Entry Point:** The main script to start training processes.
* **Argument Parsing:** Handles command-line arguments to specify config, downstream tasks, W&B mode, and number of rows.
* **Configuration Loading:** Reads and processes `config.yaml`.
* **Trainer Invocation:** Initializes and calls the `train()` method of `HFTrainer` for each specified downstream task.

---

### `trainers/train.py`

This file contains the core logic for training models.

* **`set_seed(seed)`:** Utility function to set random seeds for Python, NumPy, and PyTorch for reproducibility.
* **`CustomDataset(Dataset)`:** A PyTorch `Dataset` class to handle tokenized inputs and labels.
* **`FocalLossTrainer(Trainer)`:** A custom Hugging Face `Trainer` that implements Focal Loss for binary or multi-label classification. This is used if `loss_fn: focal_loss` is specified in the config.
* **`BaseTrainer(ABC)`:** An abstract base class defining the interface for trainers. It handles basic setup like loading environment variables and setting seeds. It also includes a `generate_report` method that orchestrates report creation using `reporting.py`.
* **`HFTrainer(BaseTrainer)`:** The main training class.
    * **Initialization:** Sets up configuration for a specific downstream task.
    * **Data Handling:**
        * `get_data()`: Loads data from CSV, Parquet, or JSON files.
        * `column_stratification_for_multilabel()` & `get_splits_for_multilabel()`: Implements iterative stratification for multi-label data splitting.
        * `get_splits_for_multiclass()`: Implements standard stratified splitting for multi-class/binary data.
        * `split_data()`: Orchestrates data splitting based on `problem_type`.
        * `preprocess_data()`: Main data preprocessing pipeline. It loads data, handles custom preprocessors (if specified in config), splits data, tokenizes text using `AutoTokenizer`, and creates `CustomDataset` instances for train, validation, and test sets.
    * **Model Initialization (`init_model()`):** Loads a pre-trained sequence classification model using `AutoModelForSequenceClassification` based on `base_model_path` and `problem_type`. Configures `id2label` and `label2id` mappings. Optionally freezes base model layers.
    * **W&B Initialization (`init_wandb()`):** Logs in to W&B, initializes a run (can resume existing runs), watches the model, and logs configuration details and dataset statistics.
    * **Metrics Computation (`compute_metrics_for_multilabel_multiclass()`, `compute_metrics()`):** Calculates evaluation metrics (accuracy, precision, recall, F1, Hamming loss, Jaccard score, etc.) during evaluation.
    * **Prediction (`get_prediction_table()`):** Generates predictions and probabilities for a given dataset split and logs them as W&B artifacts (Parquet files).
    * **Training (`train_model()`):**
        * Sets up `TrainingArguments` and `EarlyStoppingCallback`.
        * Initializes the appropriate Hugging Face `Trainer` (standard or `FocalLossTrainer`).
        * Starts training, potentially resuming from a checkpoint.
        * Evaluates the model on train, validation, and test sets after training.
        * Logs evaluation reports (including classification reports as tables) and prediction tables to W&B.
    * **Main Training Flow (`train()`):** Orchestrates the entire process: `preprocess_data` -> `init_model` -> `init_wandb` -> `train_model` -> `generate_report`.

---

### `evaluators/multilabel_classifier_evaluation.py`

This module provides tools for in-depth evaluation of multi-label classification models, primarily used by `reporting.py` to compare different models.

* **`EvalCache`:**
    * Implements a disk-based caching mechanism using `diskcache`.
    * Generates cache keys based on model state and input data hash to store and retrieve evaluation results, avoiding re-computation.
* **`MultiLabelEval`:**
    * **Initialization:** Takes a model, tokenizer, data (train, val, test DataFrames), text/label column names, thresholds, top-k values, and other evaluation parameters.
    * `_get_prediction_table()`: Generates predictions (probabilities) for a given DataFrame.
    * `_convert_to_haystack_format()`: Converts predictions and labels into Haystack `Document` format for Haystack-based evaluation metrics.
    * `_generate_metrics()`: Computes a suite of metrics based on either thresholds or top-k predictions. This includes scikit-learn metrics (precision, recall, F1, Jaccard) and Haystack metrics (DocumentRecall, MRR, NDCG, MAP).
    * `_plot_metrics()`: Generates and saves line plots for metrics against thresholds or top-k values. (Used when `evaluate()` is called directly).
    * `_evaluate_split()`: Evaluates a specific data split (train/test) and saves results to disk. (Used when `evaluate()` is called directly).
    * `evaluate()`: Runs evaluation for train and test splits and saves all outputs (inferences, metrics, plots) to a specified directory.
    * `_generate_classification_report()`: Creates a detailed classification report (precision, recall, F1 per label) using `sklearn.metrics.classification_report`.
    * `_evaluate_split_inplace()`: Evaluates a split and returns results in memory, using caching if enabled.
    * `evaluate_in_memory()`: Main method used by `reporting.py`. Evaluates train, validation (if available), and test sets, returning all results (probabilities, metrics, classification reports) in a dictionary structure.
    * `plot_comparision()` (static): Generates comparative line plots for multiple models across different metrics and data splits. Results can be saved to disk or returned as plot objects.
    * `plot_metrics_comparision_with_splits()` (static): Generates comparative bar plots for metrics, grouping by data split and showing different models. Uses hatch patterns for better accessibility. Results can be saved or returned.

---

### `reporting.py`

This module is responsible for creating comprehensive Excel reports comparing different W&B runs.

* **`create_overview_tab()`:** Creates the first "Overview" sheet in the Excel report, listing the W&B runs being compared with their names, descriptions, start times, and W&B URLs.
* **`compare_multiple_dicts()`:** A utility to compare multiple dictionaries (like W&B configs) and identify differing key-value pairs.
* **`get_all_keys()`, `get_nested_value()`, `convert_value()`:** Helper functions for `compare_multiple_dicts`.
* **`add_diff_hyperparameter()`:** Adds a "diff_hyperparameters" sheet to the report, showing a table of hyperparameters that differ across the compared W&B runs.
* **`get_loss_history_plot()`:** Fetches training and evaluation loss history from W&B runs and generates a matplotlib plot comparing them.
* **`add_loss_comparision()`:** Adds a "loss_comparison" sheet with the embedded loss history plot.
* **`get_model_from_wandb_inplace()`:** Downloads model artifacts and loads models/tokenizers from W&B runs.
* **`write_dataframe_to_excel()`:** Utility to write pandas DataFrames to an Excel sheet with good formatting (borders, header styles, merged cells for multi-index).
* **`add_classifcation_report_to_excel()`:** Adds a "classification_report" sheet containing detailed per-class metrics for each model and data split.
* **`add_metrics_comparison()`:** The main function to generate comprehensive metric comparison.
    * Fetches models and tokenizers for new and previous runs.
    * Uses `MultiLabelEval.evaluate_in_memory()` to get evaluation results for all models.
    * Calls `MultiLabelEval.plot_comparision()` to generate plots.
    * Creates three sheets:
        * "metric_comp_plots": Embeds line plots comparing metrics across different threshold/k values for each data split.
        * "split_based_comp_plots": Embeds bar plots comparing metrics across data splits for fixed threshold/k values.
        * "metrics_df_data": Contains the raw metric dataframes used for plotting.
    * Calls `add_classifcation_report_to_excel()` to add the classification report sheet.
* **`add_plots_to_worksheet()`:** Recursively adds matplotlib plot objects (from `MultiLabelEval`) to an Excel worksheet, organizing them with breadcrumb titles.
* **`calculate_rows_for_image()`, `get_column_letter()`:** Utility functions for image placement in Excel.

---

### `custom_preprocessors/preprocess.py`

Contains custom functions for data preprocessing tailored to specific datasets.

* **`preprocess_raw_json_training_file_mc_ml()`:** Example preprocessor for TDAMM data. Cleans labels, preprocesses text, and uses `MultiLabelBinarizer` to convert labels to one-hot encoded format.
* **`tdamm_preprocess()`:** A wrapper function that uses `preprocess_raw_json_training_file_mc_ml` and then splits the data into train, validation, and test sets. It's designed to be callable by `HFTrainer` when specified in `config.yaml` (e.g., `custom_preprocessor: custom_preprocessors.preprocess.tdamm_preprocess`).

---

## Dependencies

Key libraries used in this project include:

* **PyTorch:** For building and training neural networks.
* **Hugging Face Transformers:** For pre-trained models, tokenizers, and the `Trainer` API.
* **scikit-learn:** For data splitting, metrics, and multi-label binarization.
* **scikit-multilearn:** For iterative multi-label data splitting.
* **Pandas:** For data manipulation.
* **NumPy:** For numerical operations.
* **Weights & Biases (wandb):** For experiment tracking and logging.
* **Openpyxl:** For creating and manipulating Excel files (reports).
* **Matplotlib & Seaborn:** For generating plots.
* **Haystack (farm-haystack):** For advanced document retrieval metrics.
* **DiskCache:** For caching evaluation results.
* **python-dotenv:** For loading environment variables.
* **PyYAML:** For parsing YAML configuration files.

All dependencies are listed in `requirements.txt`.

---

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to existing styling and includes relevant tests.
