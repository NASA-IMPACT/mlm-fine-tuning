import hashlib
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import diskcache
import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import transformers
from haystack import Document as HaystackDocument
from haystack.components.evaluators import (
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentNDCGEvaluator,
    DocumentRecallEvaluator,
)
from sklearn.metrics import (
    classification_report,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class EvalCache:
    def __init__(self, cache_dir="eval_cache"):
        """Initialize EvalCache with diskcache."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = diskcache.Cache(cache_dir)  # Initialize the diskcache

    def _generate_cache_key(self, model, df):
        """Generates a unique hash based on model state and dataframe content."""
        # Hash model state dict
        model_state_hash = hashlib.md5(
            str(model.state_dict()).encode("utf-8"),
        ).hexdigest()

        # Hash input data (text + labels)
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes(),
        ).hexdigest()

        # Combine both hashes for a unique cache key
        return f"{model_state_hash}_{data_hash}"

    def _get_cache_key(self, model, df):
        """Generates a unique cache key using the combined model and data hashes."""
        return self._generate_cache_key(model, df)

    def load_cache(self, model, df):
        """Loads cached result if available."""
        cache_key = self._get_cache_key(model, df)
        if cache_key in self.cache:
            return self.cache[cache_key]  # Retrieve from diskcache
        return None

    def save_cache(self, model, df, result):
        """Saves result to disk with a unique cache key."""
        cache_key = self._get_cache_key(model, df)
        self.cache[cache_key] = result  # Store the result in diskcache


class MultiLabelEval:
    def __init__(
        self,
        model: Union[
            transformers.modeling_utils.PreTrainedModel,
            torch.nn.modules.module.Module,
        ],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        text_col: str,
        label_cols: List[str],
        train_data_df: pd.DataFrame = None,
        val_data_df: pd.DataFrame = None,
        test_data_df: pd.DataFrame = None,
        thresholds: List[float] = [0.3, 0.5, 0.7],
        top_ks: List[int] = [1, 3, 5],
        metrics: List[str] = [
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "weighted_jaccard_similarity",
            "haystack_document_recall",
            "haystack_document_mrr",
            "haystack_document_ndcg",
            "haystack_document_map",
        ],
        max_length: int = 512,
        use_cache: bool = True,
        model_name: str = None,
        n_rows: int = None,
    ):
        """
        Arguments:
            model: MultiLabel Classification model
            tokenizer: Tokenizer used to tokenize the text
            train_data_df: Training data
            test_data_df: Test data
            text_col: Column name containing the text
            label_cols: List of column names containing the labels (column name is expected to be the name of the label)
            thresholds: List of thresholds to use for classification
            top_ks: List of top_k values to use for classification
            max_length: Maximum length of the input text / contect size
            n_rows: Number of rows to use for training and testing. if left None all the data will be used
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_data_df = train_data_df
        self.val_data_df = val_data_df
        self.test_data_df = test_data_df
        self.text_col = text_col
        self.label_cols = label_cols
        self.thresholds = thresholds
        self.top_ks = top_ks
        self.metrics = metrics
        self.max_length = max_length
        self.use_cache = use_cache
        self.model_name = model_name
        self.n_rows = n_rows

        if self.n_rows:
            self.train_data_df = self.train_data_df.head(self.n_rows)
            self.val_data_df = self.val_data_df.head(self.n_rows)
            self.test_data_df = self.test_data_df.head(self.n_rows)

        self.metric_for_k_values = {"k": []}
        self.metric_for_threshold_values = {"threshold": []}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    def has_real_id2label(self):
        """
        Returns True if config.id2label exists and is NOT just the default
        {'0': 'LABEL_0', '1': 'LABEL_1', …} mapping.
        """
        if not hasattr(self.model.config, "id2label") or not self.model.config.id2label:
            return False

        # Build the “generic” pattern list
        default_labels = {i: f"LABEL_{i}" for i in range(self.model.config.num_labels)}

        # If it matches exactly the default, treat as “not set”
        if self.model.config.id2label == default_labels:
            return False

        # Otherwise, assume the user provided a custom map
        return True

    def _get_prediction_table(self, df: pd.DataFrame, batch_size: int = 32):
        """
        Generates a prediction table for the given DataFrame using the model.
        Args:
            df (pd.DataFrame): The input DataFrame containing the text and labels.
            batch_size (int, optional): The size of the batches to process. Defaults to 32.
        Returns:
            pd.DataFrame: A DataFrame containing the text, predicted probabilities, and expected labels.
        Notes:
            - The method sets the model to evaluation mode.
            - The input texts are tokenized and processed in batches.
            - The model's outputs are converted to probabilities using the sigmoid function.
            - The resulting probabilities and expected labels are stored in a DataFrame.
        """
        self.model.eval()
        # Get label columns from model config
        if self.has_real_id2label():
            label_cols = [
                label for idx, label in sorted(self.model.config.id2label.items())
            ]
        else:
            # raise ValueError(
            #     "Model config is missing 'id2label'. Please provide label columns.",
            # )
            warnings.warn(
                "Model config.id2label is missing or still set to generic 'LABEL_i'. Please provide a custom id2label mapping. Currently using the label_cols provided in the constructor.",
            )
            label_cols = self.label_cols

        num_batches = (len(df) + batch_size - 1) // batch_size
        all_probablities = []

        for i in tqdm(range(num_batches)):
            batch_texts = list(df[self.text_col][i * batch_size : (i + 1) * batch_size])
            batch_expected = df[self.label_cols][i * batch_size : (i + 1) * batch_size]
            # add missing columns
            for col in label_cols:
                if col not in batch_expected.columns:
                    batch_expected[col] = 0.0
            # reorder the columns
            batch_expected = batch_expected[label_cols]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                try:
                    logits = outputs.logits
                except:
                    logits = outputs["logits"]

            probabilities = torch.sigmoid(logits.clone().detach()).cpu().numpy()
            probabilities_df = pd.DataFrame(probabilities, columns=label_cols)
            #
            batch_df = pd.DataFrame(
                {
                    self.text_col: batch_texts,
                    "probability": probabilities_df.to_dict(orient="records"),
                    "label": batch_expected.to_dict(orient="records"),
                },
            )
            all_probablities.append(batch_df)

        predictions_df = pd.concat(all_probablities, ignore_index=True)
        return predictions_df

    def _convert_to_haystack_format(
        self,
        label_list: List[List[int]],
        pred_list: List[List[int]],
    ) -> Tuple[List[List[HaystackDocument]], List[List[HaystackDocument]]]:
        """
        Converts label and prediction lists to Haystack format.
        This method takes in a list of true labels and a list of predicted labels,
        and converts them into a format compatible with Haystack's evaluation
        framework. Each label and prediction is converted to uppercase and stripped
        of leading/trailing whitespace. Only labels with a value of 1 are included
        in the final lists.
        Args:
            label_list (List[List[int]]): A list of lists containing true labels.
            pred_list (List[List[int]]): A list of lists containing predicted labels.
        Returns:
            Tuple[List[List[HaystackDocument]], List[List[HaystackDocument]]]:
            A tuple containing two lists of lists of HaystackDocument objects.
            The first list contains the ground truth documents, and the second
            list contains the retrieved documents.
        """
        ground_truth_documents = []
        retrieved_documents = []

        for label_, pred_ in zip(label_list, pred_list):
            targ_list = [
                lab.upper().strip() for i, lab in zip(label_, self.label_cols) if i == 1
            ]
            pred_list = [
                lab.upper().strip() for i, lab in zip(pred_, self.label_cols) if i == 1
            ]

            gt_labels = [HaystackDocument(content=ctx) for ctx in targ_list]
            ground_truth_documents.append(gt_labels)
            pred_labels = [HaystackDocument(content=ctx) for ctx in pred_list]
            retrieved_documents.append(pred_labels)

        return ground_truth_documents, retrieved_documents

    def _generate_metrics(self, prob_df: pd.DataFrame, basis: str) -> pd.DataFrame:
        """
        Generate evaluation metrics for multilabel classification.
        Parameters:
        -----------
        prob_df : pd.DataFrame
            DataFrame containing the probabilities and target labels.
            It should have columns "probability" and "target", where "probability"
            contains the predicted probabilities and "target" contains the true labels.
        basis : str
            The basis for generating metrics. It can be either "threshold" or "k".
            If "threshold", metrics are calculated based on different threshold values.
            If "k", metrics are calculated based on top-k values.
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the calculated metrics for each threshold or top-k value.
            The metrics include weighted precision, weighted recall, weighted F1 score,
            weighted Jaccard similarity, and various Haystack document retrieval metrics.
        """
        metric_dict = defaultdict(list)
        label_list = prob_df["label"].apply(lambda x: list(x.values())).tolist()
        if basis == "threshold":
            threshold_values = self.thresholds
        elif basis == "k":
            threshold_values = self.top_ks

        for th in threshold_values:
            if basis == "threshold":
                pred_list = (
                    prob_df["probability"]
                    .apply(lambda x: [int(i >= th) for i in x.values()])
                    .tolist()
                )

            elif basis == "k":
                pred_list = (
                    prob_df["probability"]
                    .apply(pd.Series)
                    .apply(
                        lambda row: (row >= row.nlargest(th).min()).astype(int),
                        axis=1,
                    )
                    .values.tolist()
                )

            sklearn_metrics = {
                "weighted_precision": precision_score,
                "weighted_recall": recall_score,
                "weighted_f1": f1_score,
                "weighted_jaccard_similarity": jaccard_score,
            }

            metric_dict[basis].append(th)
            for metric_name, metric_func in sklearn_metrics.items():
                if metric_name not in self.metrics:
                    continue
                score = metric_func(
                    label_list,
                    pred_list,
                    average="weighted",
                    zero_division=0,
                )
                metric_dict[metric_name].append(score)

            haystack_gt_docs, haystack_pred_docs = self._convert_to_haystack_format(
                label_list,
                pred_list,
            )
            haystack_metrics = {
                "haystack_document_recall": DocumentRecallEvaluator,
                "haystack_document_mrr": DocumentMRREvaluator,
                "haystack_document_ndcg": DocumentNDCGEvaluator,
                "haystack_document_map": DocumentMAPEvaluator,
            }

            for evaluator_str, evaluator in haystack_metrics.items():
                if evaluator_str not in self.metrics:
                    continue
                score = (
                    evaluator()
                    .run(
                        ground_truth_documents=haystack_gt_docs,
                        retrieved_documents=haystack_pred_docs,
                    )
                    .get("score")
                )
                metric_dict[evaluator_str].append(score)

        return pd.DataFrame(metric_dict)

    def _plot_metrics(self, df: pd.DataFrame, split: str, dir: str):
        """
        Plots and saves line plots for various metrics from a DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame containing the metrics to be plotted. It should have columns
                           representing different metrics and either 'threshold' or 'k' as a basis for plotting.
        split (str): A string indicating the dataset split (e.g., 'train', 'validation', 'test') for the plot title.
        dir (str): Directory path where the plots will be saved.

        The function creates a subdirectory named 'line_plots' within the specified directory and saves the plots
        as PNG files in this subdirectory. Each plot shows the relationship between the basis (either 'threshold'
        or 'k') and a metric from the DataFrame.
        """
        line_plot_dir = os.path.join(dir, "line_plots")
        os.makedirs(line_plot_dir, exist_ok=True)

        basis = None
        if "threshold" in df.columns:
            basis = "threshold"
        elif "k" in df.columns:
            basis = "k"

        comparision_metrics = [i for i in df.columns if i not in [basis]]

        for metric_name in comparision_metrics:
            plt.figure(figsize=(8, 6))

            plt.plot(df[basis], df[metric_name], marker="o")

            plt.xlabel(basis)
            plt.ylabel(metric_name)
            plt.title(f"{basis} vs {metric_name} for {split} dataset")
            plt.legend(loc="best")
            # Save the plot to the specified file location
            plot_file_path = os.path.join(line_plot_dir, f"{metric_name}.png")
            plt.savefig(plot_file_path)
            # Close the plot to free up memory
            plt.close()

    def _evaluate_split(self, split: str, output_dir: str):
        """
        Evaluate the model on a given data split and save the results.

        This method performs the following steps:
        1. Creates the output directory if it does not exist.
        2. Selects the appropriate data frame based on the split (train or test).
        3. Makes predictions and generates a probability table.
        4. Saves the inference results to a parquet file.
        5. Generates evaluation metrics based on 'k' values and threshold values.
        6. Saves the generated metrics to parquet files.
        7. Plots the metrics and saves the plots.

        Args:
            split (str): The data split to evaluate ('train' or 'test').
            output_dir (str): The directory where the output files will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        if split == "train":
            df = self.train_data_df
        elif split == "test":
            df = self.test_data_df

        # make inference
        prob_df = self._get_prediction_table(df)

        # save the inferences
        inference_dir = os.path.join(output_dir, "inference")
        os.makedirs(inference_dir, exist_ok=True)
        prob_df.to_parquet(os.path.join(inference_dir, f"{split}_prob_df.parquet"))

        # generate metrics
        metric_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metric_dir, exist_ok=True)
        k_dir = os.path.join(metric_dir, "k")
        os.makedirs(k_dir, exist_ok=True)
        threshold_dir = os.path.join(metric_dir, "threshold")
        os.makedirs(threshold_dir, exist_ok=True)
        self.metric_for_k_values = self._generate_metrics(prob_df, basis="k")
        self.metric_for_threshold_values = self._generate_metrics(
            prob_df,
            basis="threshold",
        )

        # save the metrics
        self.metric_for_k_values.to_parquet(
            os.path.join(k_dir, f"{split}_k_metrics.parquet"),
        )
        self.metric_for_threshold_values.to_parquet(
            os.path.join(threshold_dir, f"{split}_threshold_metrics.parquet"),
        )

        # plot mertics for for basis
        self._plot_metrics(self.metric_for_k_values, split, k_dir)
        self._plot_metrics(self.metric_for_threshold_values, split, threshold_dir)

    def evaluate(self, output_dir: str = "./multi_label_evaluations"):
        """
        Evaluates the model on the training and test datasets and saves the evaluation results to the specified output directory.

        Args:
            output_dir (str): The directory where the evaluation results will be saved. Defaults to "./multi_label_evaluations".

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
        if self.train_data_df is not None:
            self._evaluate_split("train", os.path.join(output_dir, "train"))
        if self.test_data_df is not None:
            self._evaluate_split("test", os.path.join(output_dir, "test"))

    def _generate_classification_report(
        self,
        prob_df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        # Step 1: Collect all unique labels across the dataset
        all_labels = set()
        for label_dict in prob_df["label"]:
            all_labels.update(label_dict.keys())
        label_names = sorted(all_labels)

        # Step 2: Convert dict columns to DataFrame columns, ensuring consistent label order
        y_true = pd.DataFrame(
            [
                {label: d.get(label, 0.0) for label in label_names}
                for d in prob_df["label"]
            ],
        )
        y_probs = pd.DataFrame(
            [
                {label: d.get(label, 0.0) for label in label_names}
                for d in prob_df["probability"]
            ],
        )

        # Step 3: Apply threshold to get binary predictions
        y_pred = (y_probs >= threshold).astype(int)

        # Step 4: Generate classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        # Step 5: Convert to DataFrame
        report_df = pd.DataFrame(report).transpose().round(4)

        return report_df

    def _evaluate_split_inplace(self, split: str):
        """
        Evaluate the model on a specified data split (train, val, or test) and return the results.
        This method performs the following steps:
        1. Determines the appropriate dataset (train, validation, or test) based on the `split` argument.
        2. Checks if cached results are available for the specified model and dataset. If available,
           it uses the cached results to avoid redundant computations.
        3. If no cache is available, it generates predictions and computes evaluation metrics.
        4. Saves the computed results to the cache for future use.
        Args:
            split (str): The data split to evaluate. Must be one of "train", "val", or "test".
        Returns:
            dict: A dictionary containing the following keys:
                - "prob_df": A DataFrame containing the predicted probabilities for the dataset.
                - "k_metrics": Metrics computed based on top-k predictions.
                - "threshold_metrics": Metrics computed based on threshold-based predictions.
        Raises:
            ValueError: If the `split` argument is not one of "train", "val", or "test".
        """
        result = {}
        if split == "train":
            df = self.train_data_df

        elif split == "val":
            df = self.val_data_df

        elif split == "test":
            df = self.test_data_df

        cache = EvalCache()
        if self.use_cache:
            # Check if cache exists
            cached_result = cache.load_cache(self.model, df)
            if cached_result is not None:
                print(
                    f"✅ Using cached results for {self.model_name} on {split} dataset.",
                )
                result = cached_result.copy()
                return result

        # Generate result if no cache available
        print(f"🔄 Generating Result... for {self.model_name} on {split} dataset.")

        # make inference
        prob_df = self._get_prediction_table(df)
        result["prob_df"] = prob_df

        # generate metrics
        self.metric_for_k_values = self._generate_metrics(prob_df, basis="k")
        self.metric_for_threshold_values = self._generate_metrics(
            prob_df,
            basis="threshold",
        )
        result["k_metrics"] = self.metric_for_k_values
        result["threshold_metrics"] = self.metric_for_threshold_values
        result["classification_report"] = self._generate_classification_report(
            prob_df,
            threshold=0.5,
        )

        # Save predictions to cache
        cache.save_cache(self.model, df, result.copy())

        return result

    def evaluate_in_memory(self):
        """
        Evaluates the model's performance on the train, validation, and test datasets
        (if available) and returns the results.

        This method evaluates the model in memory by calling the `_evaluate_split_inplace`
        method for each dataset split (train, validation, and test) that is not `None`.

        Returns:
            dict: A dictionary containing evaluation results for each dataset split.
                  The keys are "train", "val", and "test", and the values are the
                  corresponding evaluation results. If a dataset split is not available,
                  it will not be included in the result dictionary.
        """
        result = {}

        if self.train_data_df is not None:
            result["train"] = {}
            result["train"] = self._evaluate_split_inplace("train")

        if self.val_data_df is not None:
            result["val"] = {}
            result["val"] = self._evaluate_split_inplace("val")

        if self.test_data_df is not None:
            result["test"] = {}
            result["test"] = self._evaluate_split_inplace("test")

        return result

    @staticmethod
    def plot_comparision(
        results: Dict,
        project_name: str,
        output_dir: str = None,
        font_size: int = 12,
        ledgend_n_cols: int = 2,
        metric_dfs: Dict = None,
    ):
        """
        Plots comparison metrics for multi-label classification models and either saves the plots to the specified
        directory or returns them as a dictionary.

        Args:
            results (Dict): A dictionary containing evaluation results for multiple models.
                            The structure should be:
                            {
                                "model_name": {
                                    "data_split": {
                                        "prob_df": DataFrame,
                                        "k_metrics": DataFrame,
                                        "threshold_metrics": DataFrame
                                    }
                                }
                            }
            project_name (str): Name of the project, used for labeling and organizing output.
            output_dir (str, optional): Directory where the plots and metrics will be saved. If None, function
                                        returns a dictionary containing the plots instead of saving to files.
            font_size (int, optional): Font size for plot labels, titles, and legends. Defaults to 12.
            ledgend_n_cols (int, optional): Number of columns in the legend. Defaults to 2.
            metric_dfs (Dict, optional): Precomputed metric DataFrames for "k" and "threshold" metrics. If None,
                                        the function will compute these from the `results` dictionary. Defaults to None.

        Returns:
            If output_dir is None, returns a dictionary containing the plots with the structure:
            {
                "split_name": {
                    "metrics": {
                        "basis_name": {
                            "line_plots": {
                                "metric_name_comparision": matplotlib_figure_object
                            }
                        }
                    }
                },
                "split_based_comparision": MultiLabelEval.plot_metrics_comparision_with_splits_return_value
            }
            If output_dir is not None, returns None but saves all plots to the specified directory.

        Behavior:
            - Transforms the `results` dictionary into a plotting-friendly structure if `metric_dfs` is not provided.
            - If output_dir is not None, saves intermediate DataFrames (`k_metrics` and `threshold_metrics`) as parquet
            files and saves plots as PNG files.
            - If output_dir is None, stores the plots in a dictionary structure.
            - Generates line plots for each metric, comparing models across different data splits.
            - Calls `MultiLabelEval.plot_metrics_comparision_with_splits` to generate additional split-based comparisons.

        Note:
            - The function assumes that `results` contains DataFrames with specific keys (`prob_df`, `k_metrics`,
            `threshold_metrics`).
            - When output_dir is None, the `MultiLabelEval.plot_metrics_comparision_with_splits` method must support
            a return mode that returns plots instead of saving them.

        Raises:
            - Any exceptions related to file I/O or DataFrame operations will propagate to the caller.
        """
        # Initialize the plots dictionary if output_dir is None
        plots_dict = (
            {"metric_comp_plots": {}, "metrics_data": {}}
            if output_dir is None
            else None
        )

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        if metric_dfs is None:
            # Transforming the results data structure to a plotting friendly data structure
            dfs_k_metrics = []
            dfs_threshold_metrics = []
            for model_name, data_splits in results.items():
                for split, metrics in data_splits.items():
                    inferences_df = metrics["prob_df"].copy()

                    if output_dir is not None:
                        inference_dir = os.path.join(output_dir, split, "inference")
                        os.makedirs(inference_dir, exist_ok=True)
                        inferences_df.to_parquet(
                            os.path.join(
                                inference_dir,
                                f"{model_name}_{split}_prob_df.parquet",
                            ),
                        )

                    k_metrics_df = metrics["k_metrics"].copy()
                    threshold_metrics_df = metrics["threshold_metrics"].copy()

                    k_metrics_df["model"] = model_name
                    k_metrics_df["data_split"] = split
                    dfs_k_metrics.append(k_metrics_df)

                    threshold_metrics_df["model"] = model_name
                    threshold_metrics_df["data_split"] = split
                    dfs_threshold_metrics.append(threshold_metrics_df)

            # Concatenate all DataFrames
            dfs_k_metrics = pd.concat(dfs_k_metrics, ignore_index=True)
            dfs_threshold_metrics = pd.concat(dfs_threshold_metrics, ignore_index=True)
            metric_dfs = {"k": dfs_k_metrics, "threshold": dfs_threshold_metrics}
        else:
            pass

        markers = ["o", "s", "^", "D", "v", "p", "*", "x", "+"]
        linestyles = ["-", "--", "-.", ":"]

        for basis, metric_df in metric_dfs.items():
            if output_dir is not None:
                metric_df.to_parquet(f"{output_dir}/{basis}_all_metrics.parquet")
            else:
                plots_dict["metrics_data"][basis] = metric_df

            for split in metric_df["data_split"].unique():
                if output_dir is None and split not in plots_dict["metric_comp_plots"]:
                    plots_dict["metric_comp_plots"][split] = {}

                if (
                    output_dir is None
                    and basis not in plots_dict["metric_comp_plots"][split]
                ):
                    plots_dict["metric_comp_plots"][split][basis] = {}

                df_split = metric_df[metric_df["data_split"] == split]
                comparision_metrics = [
                    i
                    for i in df_split.columns
                    if i not in [basis, "model", "data_split"]
                ]

                for metric_name in comparision_metrics:
                    fig = plt.figure(figsize=(14, 6))
                    for _i, model in enumerate(df_split["model"].unique()):
                        dx = df_split[df_split["model"] == model]
                        marker = markers[_i % len(markers)]
                        linestyle = linestyles[_i % len(linestyles)]
                        plt.plot(
                            dx[basis],
                            dx[metric_name],
                            marker=marker,
                            linestyle=linestyle,
                            label=model,
                        )
                    plt.title(
                        f"{basis} vs {metric_name}",
                        fontsize=font_size + 2,
                        fontweight="bold",
                    )
                    plt.xlabel(basis, fontsize=font_size, fontweight="bold")
                    plt.ylabel(metric_name, fontsize=font_size, fontweight="bold")

                    plt.xticks(fontsize=font_size - 2)
                    plt.yticks(fontsize=font_size - 2)

                    legend = plt.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.15),  # Place legend at bottom
                        ncol=ledgend_n_cols,  # Multiple columns if necessary
                        fontsize=font_size,
                        frameon=False,
                    )
                    legend.set_title(None)  # Remove legend title
                    plt.tight_layout()

                    title = metric_name + "_comparision"

                    if output_dir is not None:
                        # Save the plot to the specified file location
                        file_loc = os.path.join(
                            output_dir,
                            split,
                            "metrics",
                            basis,
                            "line_plots",
                        )
                        os.makedirs(file_loc, exist_ok=True)
                        plot_file_path = os.path.join(file_loc, f"{title}.png")
                        plt.savefig(plot_file_path)
                        plt.close()
                    else:
                        # Store the figure in the plots dictionary
                        plots_dict["metric_comp_plots"][split][basis][title] = fig

        # Call plot_metrics_comparision_with_splits for split-based comparisons
        if output_dir is not None:
            split_based_com_output_dir = os.path.join(
                output_dir,
                "split_based_comparision",
            )
        else:
            split_based_com_output_dir = None

        plots_dict[
            "split_based_comparision"
        ] = MultiLabelEval.plot_metrics_comparision_with_splits(
            results=None,
            font_size=font_size,
            output_dir=split_based_com_output_dir,
            project_name=project_name,
            ledgend_n_cols=ledgend_n_cols,
            metric_dfs=metric_dfs,
            save_metrics=False,
        )

        return plots_dict if output_dir is None else None

    @staticmethod
    def generate_model_to_hatch(models, hatch_patterns):
        return {
            model: hatch_patterns[i % len(hatch_patterns)]
            for i, model in enumerate(models)
        }

    @staticmethod
    def plot_metrics_comparision_with_splits(
        results: Dict,
        project_name: str,
        model_order: List[str] = None,
        output_dir: str = None,
        font_size: int = 12,
        ledgend_n_cols: int = 2,
        metric_dfs: Dict = None,
        save_metrics: bool = True,
    ):
        """
        Plots and optionally saves bar charts comparing metrics across data splits for multiple models.
        If output_dir is None, returns a dictionary of plots instead of saving them.

        Args:
            results (Dict): A dictionary containing evaluation results for multiple models
                and data splits. The structure is expected to be:
                {
                    "model_name": {
                        "data_split": {
                            "k_metrics": pd.DataFrame,
                            "threshold_metrics": pd.DataFrame
                        }
                    }
                }
            project_name (str): The name of the project, used in plot titles.
            model_order (List[str], optional): A list specifying the order of models
                to display in the plots. If None, the order is determined by the data.
                Defaults to None.
            output_dir (str, optional): The directory where the plots and data files
                will be saved. If None, plots are returned instead. Defaults to None.
            font_size (int, optional): Font size for plot labels, ticks, and titles.
                Defaults to 12.
            ledgend_n_cols (int, optional): Number of columns in the legend. Defaults to 2.
            metric_dfs (Dict, optional): Precomputed metric DataFrames for "k" and
                "threshold" metrics. If provided, it should be a dictionary with keys
                "k" and "threshold" containing the respective DataFrames. If None, the
                DataFrames are generated from the `results` argument. Defaults to None.
            save_metrics (bool, optional): Whether to save metric DataFrames when output_dir
                is provided. Ignored when output_dir is None. Defaults to True.

        Returns:
            Dict or None: If output_dir is None, returns a dictionary with structure:
                {
                    "k": {
                        "metric_name": {
                            "k_value": plot_figure
                        }
                    },
                    "threshold": {
                        "metric_name": {
                            "threshold_value": plot_figure
                        }
                    }
                }
                If output_dir is not None, returns None and saves plots to disk.

        Notes:
            - The function uses hatch patterns to differentiate models in the plots,
                improving accessibility for colorblind users.
            - The plots include value annotations for each bar.
            - When saving, the function ensures that the output directory structure is created
                before saving files.
        """
        # Initialize return dictionary if not saving to disk
        plots_dict = {"k": {}, "threshold": {}} if output_dir is None else None

        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        if metric_dfs is None:
            # transforming the results data structure to a plottting friendly data structure
            dfs_k_metrics = []
            dfs_threshold_metrics = []

            if model_order is None:
                model_order = list(results.keys())

            for model_name, data_splits in results.items():
                for split, metrics in data_splits.items():
                    k_metrics_df = metrics["k_metrics"].copy()
                    threshold_metrics_df = metrics["threshold_metrics"].copy()

                    k_metrics_df["model"] = model_name
                    k_metrics_df["data_split"] = split
                    dfs_k_metrics.append(k_metrics_df)

                    threshold_metrics_df["model"] = model_name
                    threshold_metrics_df["data_split"] = split
                    dfs_threshold_metrics.append(threshold_metrics_df)

            # Concatenate all DataFrames
            dfs_k_metrics = pd.concat(dfs_k_metrics, ignore_index=True)
            dfs_threshold_metrics = pd.concat(dfs_threshold_metrics, ignore_index=True)
        else:
            dfs_k_metrics = metric_dfs.get("k")
            dfs_threshold_metrics = metric_dfs.get("threshold")

        for basis, metric_df in {
            "k": dfs_k_metrics,
            "threshold": dfs_threshold_metrics,
        }.items():
            if output_dir is not None:
                dir_to_save_data = os.path.join(output_dir, basis)
                os.makedirs(dir_to_save_data, exist_ok=True)
                if save_metrics:
                    metric_df.to_parquet(
                        f"{dir_to_save_data}/{basis}_all_metrics.parquet",
                    )

            comparision_metrics = [
                i for i in metric_df.columns if i not in [basis, "model", "data_split"]
            ]
            # Define hatch patterns for each model to address colorblindness
            hatch_patterns = [
                "/",
                "\\",
                "|",
                "-",
                "+",
                "x",
                "o",
                "*",
                "O",
                "//",
                "\\\\",
                "||",
                "--",
                "++",
                "xx",
                "oo",
                "..",
                "**",
            ]
            # Create model-to-hatch mapping
            model_to_hatch = MultiLabelEval.generate_model_to_hatch(
                metric_df["model"].unique(),
                hatch_patterns,
            )

            for metric_name in comparision_metrics:
                # Initialize metric dictionary in the return structure if returning plots
                if plots_dict is not None and metric_name not in plots_dict[basis]:
                    plots_dict[basis][metric_name] = {}

                for th in metric_df[basis].unique():
                    df_th = metric_df[metric_df[basis] == th]
                    fig_width = (df_th["model"].nunique()) * (
                        df_th["data_split"].nunique()
                    ) + 2
                    plt.figure(figsize=(fig_width, 6))
                    plt.rcParams.update(
                        {
                            "axes.labelsize": font_size,  # Font size for x and y labels
                            "axes.labelweight": "bold",  # Font weight for x and y labels
                            "xtick.labelsize": font_size
                            - 2,  # Font size for x-axis ticks
                            "ytick.labelsize": font_size
                            - 2,  # Font size for y-axis ticks
                        },
                    )

                    model_order = (
                        df_th.groupby("model")[metric_name]
                        .mean()
                        .sort_values(ascending=False)
                        .index.tolist()
                    )
                    df_th["model"] = pd.Categorical(
                        df_th["model"],
                        categories=model_order,
                        ordered=True,
                    )
                    ax = sns.barplot(
                        x="data_split",
                        y=metric_name,
                        hue="model",
                        data=df_th,
                    )

                    # Extract colors assigned by Seaborn
                    handles, labels = ax.get_legend_handles_labels()
                    colors = {
                        label: handle.get_facecolor()
                        for handle, label in zip(handles, labels)
                    }

                    # Add hatch patterns to bars
                    for bars, mode_str in zip(
                        ax.containers,
                        model_order,
                    ):  # Iterate based on sorted order
                        for bar in bars:
                            bar.set_hatch(model_to_hatch[mode_str])
                            bar.set_edgecolor("white")

                    # Create custom legend patches with both color and hatch
                    legend_patches = [
                        mpatches.Patch(
                            facecolor=colors[model_str],
                            edgecolor="white",
                            hatch=model_to_hatch[model_str],
                            label=model_str,
                        )
                        for model_str in model_order
                    ]
                    ax.legend(
                        handles=legend_patches,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.15),
                        ncol=ledgend_n_cols,
                        fontsize=font_size,
                        frameon=False,
                        handleheight=2,
                        handlelength=3,
                    )

                    plt.title(
                        f"Comparision of {metric_name} at {basis}={th} for {project_name}",
                        fontsize=font_size + 2,
                        fontweight="bold",
                    )

                    # Add value labels
                    for p in ax.patches:
                        height = p.get_height()
                        if p.get_width():
                            ax.annotate(
                                f"{height:.2f}",
                                (p.get_x() + p.get_width() / 2.0, height),
                                ha="center",
                                va="bottom",
                                fontsize=font_size - 2,
                                color="black",
                                fontweight="bold",
                            )
                    plt.tight_layout()

                    # Either save the plot or add it to the return dictionary
                    if output_dir is not None:
                        dir_to_save_plot = os.path.join(dir_to_save_data, metric_name)
                        os.makedirs(dir_to_save_plot, exist_ok=True)
                        plt.savefig(
                            f"{dir_to_save_plot}/{metric_name}_{basis}_{th}.png",
                        )
                        plt.close()
                    else:
                        # Store the figure in the return dictionary
                        plots_dict[basis][metric_name][th] = plt.gcf()

        return plots_dict


def print_dict_structure(d: Dict, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key} (dict)")
            print_dict_structure(value, indent + 1)
        elif isinstance(value, list):
            print(
                "  " * indent
                + f"{key} (list[{type(value[0]).__name__}] if len(value) > 0 else 'empty list')",
            )
        else:
            print("  " * indent + f"{key} ({type(value).__name__})")


if __name__ == "__main__":
    # import sentiment analysis model and tokenizer from huggingface
    import json

    from sklearn.model_selection import train_test_split
    from tdamm.preprocess import preprocess_raw_json_training_file_mc_ml
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # get the data, split it into train and test
    with open("./tdamm/tdamm_training_data_multi_labelled.json", "r") as file:
        data = json.load(file)
    df = preprocess_raw_json_training_file_mc_ml(data).fillna(0)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("adsabs/astroBERT")
    model = AutoModelForSequenceClassification.from_pretrained("./tdamm/final_model/")

    # column name list in dataframe
    label_cols = [
        "Active Galactic Nuclei",
        "Binary Black Holes",
        "Binary Neutron Stars",
        "Binary Pulsars",
        "Burst",
        "Cataclysmic Variables",
        "Compact Binary Inspiral",
        "Continuous",
        "Cosmic Rays",
        "Exoplanets",
        "Fast Blue Optical Transients",
        "Fast Radio Bursts",
        "Gamma rays",
        "Gamma-ray Bursts",
        "Infrared",
        "Intermediate Mass",
        "Kilonovae",
        "Magnetars",
        "Neutrinos",
        "Neutron Star-Black Hole",
        "Novae",
        "Optical",
        "Pevatrons",
        "Pulsar Wind Nebulae",
        "Pulsars",
        "Radio",
        "Stellar Mass",
        "Stellar flares",
        "Stochastic",
        "SuperNovae",
        "Supermassive",
        "Supernova Remnants",
        "Ultraviolet",
        "White Dwarf Binaries",
        "X-rays",
        "non-TDAMM",
    ]
    text_column = "full_text"

    evaluator = MultiLabelEval(
        model=model,
        tokenizer=tokenizer,
        train_data_df=train_df,
        test_data_df=test_df,
        text_col=text_column,
        label_cols=label_cols,
        thresholds=[0.3, 0.5, 0.7],
        top_ks=[1, 3, 5],
        max_length=512,
        n_rows=100,  # for testing
    )

    # test = evaluator.evaluate(output_dir="./multi_label_evaluations")
    result = evaluator.evaluate_in_memory()
    print(result)
