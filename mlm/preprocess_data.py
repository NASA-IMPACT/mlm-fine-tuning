import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import diskcache

# from sentence_transformers import SentenceTransformer
import torch
from datasets import Dataset, DatasetDict, load_dataset
from nltk.metrics import edit_distance
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)

# Create a cache object
cache = diskcache.Cache("./_datacache/")


def disk_cache(func):
    def wrapper(*args, **kwargs):
        # Generate a unique key based on the function arguments
        key = hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()

        # Check if result is cached
        if key in cache:
            print("Loading from cache...")
            return cache[key]

        # Call the function and cache the result
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper


class KeywordMasking:
    @staticmethod
    def find_max_iou_edit_distance(word: str, keywords: list[str]) -> float:
        """Finds max IoU using edit distance as the difference measure."""
        max_iou = 0.0
        for keyword in keywords:
            dist = edit_distance(word, keyword)
            union = len(word) + len(keyword) - dist
            iou = (union - dist) / union if union > 0 else 0.0
            max_iou = max(max_iou, iou)
        return max_iou

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        input_text_column: str,
        key_extrator: str = "YAKE",
        mlm_probablity: float = 0.3,
        keyword_masking_probablity: float = 0.85,
        keyword_selection_percentile: float = 50.0,
        keyword_iou_threshold: float = 0.80,
    ):
        self.tokenizer = tokenizer
        self.input_text_column = input_text_column
        self.key_extrator = key_extrator
        self.mlm_probablity = mlm_probablity
        self.keyword_masking_probablity = keyword_masking_probablity
        self.keyword_selection_percentile = keyword_selection_percentile
        self.keyword_iou_threshold = keyword_iou_threshold

        if self.key_extrator == "YAKE":
            import pke

            self.extractor = pke.unsupervised.YAKE()
        else:
            raise ValueError(f"Invalid keyphrase extractor: {self.key_extrator}")

    def mask_with_keywords(self, tokenized: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """Mask the keywords in the input text."""
        # get the final probability matrix
        lm_dataset = tokenized.map(
            self.gen_prob_matrix,
            batched=True,
            num_proc=8,
            fn_kwargs={},
            desc="Gen Prob Matrix",
        )

        # use this probability matrix to mask the tokens
        lm_dataset = lm_dataset.map(
            self.gen_static_mask,
            batched=True,
            num_proc=8,
            fn_kwargs={},
            desc="Static Masking",
        )
        return lm_dataset

    def gen_static_mask(self, batch: Dict[str, Dataset]) -> Dict[str, List[Any]]:
        """Generate the masked tokens."""
        probability_matrix = torch.tensor(
            batch["probability_matrix"],
            dtype=torch.float,
        )
        input_ids = torch.tensor(batch["input_ids"])
        input_ids_clone = input_ids.clone()
        labels = input_ids.clone()

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token,
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer),
            labels.shape,
            dtype=torch.long,
        )
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # Return modified batch
        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_ids_clone": input_ids_clone,
        }

    def get_mask_candidates_YAKE_batch(
        self,
        texts: list[str],
        offset_mapping: list[Tuple[int, int]],
    ) -> torch.Tensor:
        """Get the candidate keywords for masking using YAKE."""
        keyword_mask_candidates = torch.zeros(
            (len(texts), len(offset_mapping[0])),
            dtype=torch.float,
        )  # 2D Tensor
        for _i, (text, of_map) in enumerate(zip(texts, offset_mapping)):
            self.extractor.load_document(input=text, language="en")
            self.extractor.candidate_selection(n=1)  # getting only 1-gram keywords
            self.extractor.candidate_weighting()
            total_n_keywords = len(self.extractor.weights)
            n_keywords_to_select = int(
                (self.keyword_selection_percentile * total_n_keywords) // 100,
            )

            keywords = [
                k.upper().strip()
                for k, _ in self.extractor.get_n_best(n=n_keywords_to_select)
            ]

            # use regex way
            keyword_pattern = re.compile(
                r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
                re.IGNORECASE,
            )
            is_in_keyword = torch.tensor(
                [
                    1 if re.search(keyword_pattern, text[start:end].strip()) else 0
                    for start, end in of_map
                ],
                dtype=torch.float,
            )

            # is_in_keyword = torch.tensor([text[m[0]: m[1]].upper().strip() in keywords  for m in of_map], dtype=torch.float) #uses exact match
            # is_in_keyword = torch.tensor(
            #     [
            #         self.find_max_iou_edit_distance(text[m[0] : m[1]].upper(), keywords)
            #         > self.keyword_iou_threshold
            #         for m in of_map
            #     ],
            #     dtype=torch.float,
            # )
            keyword_mask_candidates[_i, : len(is_in_keyword)] = is_in_keyword
        return keyword_mask_candidates

    def gen_prob_matrix(self, batch: Dict[str, Dataset]) -> Dict[str, torch.Tensor]:
        """Generate the probability matrix for MLM."""
        input_ids = torch.tensor(batch["input_ids"])
        labels = input_ids.clone()  # Create labels
        offset_mapping = batch["offset_mapping"]
        texts = batch[self.input_text_column]

        keyword_mask_candidates = self.get_mask_candidates_YAKE_batch(
            texts,
            offset_mapping,
        )

        # keyword based masking for self.keyword_masking_probablity of the elements, for rest of the case we will use random masking
        random_mask_candidates = torch.full(labels.shape, 1.0)
        keyword_based_prob_mask = (
            torch.rand_like(keyword_mask_candidates, dtype=torch.float)
            < self.keyword_masking_probablity
        )
        probability_matrix = (
            torch.where(
                keyword_based_prob_mask,
                keyword_mask_candidates,
                random_mask_candidates,
            )
            * self.mlm_probablity
        )

        # dotn want to mask special characters
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        return {"probability_matrix": probability_matrix}


def split_dataset(
    ds: DatasetDict,
    val_split: Optional[float],
    test_split: Optional[float],
) -> DatasetDict:
    if val_split is not None and test_split is not None:
        temp_split = ds["train"].train_test_split(
            test_size=(val_split + test_split),
            seed=42,
        )
        val_test_split = temp_split["test"].train_test_split(
            test_size=test_split / (val_split + test_split),
            seed=42,
        )

        ds = DatasetDict(
            {
                "train": temp_split["train"],
                "validation": val_test_split["train"],
                "test": val_test_split["test"],
            },
        )
    return ds


# def compute_text_level_embeddings(batch: Dict[str, List[str]], text_column: str, model: SentenceTransformer) -> Dict[str, List[List[float]]]:
#     """Computes text embeddings for a batch of text data."""
#     embeddings = model.encode(batch[text_column], convert_to_numpy=True)  # Encode text
#     batch["text_embedding"] = [emb.tolist() for emb in embeddings]  # Convert to list for Dataset compatibility
#     return batch

# def generate_text_level_embeddings(ds: DatasetDict, text_column: str) -> DatasetDict:
#     """Generates and stores text embeddings in the dataset."""
#     emb_model = SentenceTransformer("all-MiniLM-L6-v2")
#     emb_model = emb_model.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

#     ds = ds.map(
#         compute_text_level_embeddings,
#         batched=True,
#         batch_size=32,
#         fn_kwargs={"model": emb_model, "text_column": text_column},
#     )
#     return ds


def get_dataset(
    dataset_config: Dict[str, Any],
    data_src: str,
    n_rows: int,
    split: Optional[bool] = True,
):
    ds = DatasetDict(load_dataset(**dataset_config.get(data_src)))
    if n_rows:
        ds = DatasetDict(
            {split: ds[split].select(range(n_rows)) for split in ds.keys()},
        )

    if split:
        return split_dataset(
            ds,
            dataset_config["val_split"],
            dataset_config["test_split"],
        )
    else:
        return ds


def chunk_texts(
    examples: Dict[str, List[List[int]]],
    chunk_size: int = 128,
) -> Dict[str, List[List[int]]]:
    results: Dict[str, List[List[int]]] = {}
    for k in examples.keys():
        results[k] = []
        # expecting k to be values like input_ids and attention_masks
        for d in range(len(examples[k])):
            # expecting d to be values for a document
            if len(examples[k][d]) <= chunk_size:
                results[k].append(examples[k][d])
            else:
                for i in range(0, len(examples[k][d]), chunk_size):
                    results[k].append(examples[k][d][i : i + chunk_size])

    return results


def preprocess_dataset(
    input_config: Dict[str, Any],
    data_src: str,
    n_rows: Optional[int] = None,
) -> Tuple[DatasetDict, PreTrainedTokenizer, DataCollatorForLanguageModeling]:

    dataset = get_dataset(input_config["dataset"], data_src, n_rows)
    tokenizer = AutoTokenizer.from_pretrained(input_config["model"]["hf"])

    tokenized_ds = dataset.map(
        lambda examples: tokenizer(
            examples[input_config.get("dataset").get("text_column")],
            truncation=True,
        ),
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )
    lm_dataset = tokenized_ds.map(
        chunk_texts,
        batched=True,
        num_proc=4,
        fn_kwargs={"chunk_size": input_config["dataset"]["chunk_size"]},
    )
    # tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=input_config["dataset"]["mlm_probability"],
    )

    return lm_dataset, tokenizer, data_collator


@disk_cache
def preprocess_dataset_with_static_masking(
    input_config: Dict[str, Any],
    data_src: str,
    n_rows: Optional[int] = None,
) -> Tuple[DatasetDict, PreTrainedTokenizer, DataCollatorForLanguageModeling]:

    dataset = get_dataset(input_config["dataset"], data_src, n_rows, True)
    tokenizer = AutoTokenizer.from_pretrained(input_config["model"]["hf"])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize_function(examples):
        tokenized_output = tokenizer(
            examples[input_config.get("dataset").get("text_column")],
            max_length=input_config["dataset"]["chunk_size"],
            truncation=True,
            padding="max_length",  # Ensures uniform sequence length
            return_overflowing_tokens=True,
            return_length=True,
            return_offsets_mapping=True,
        )

        overflow_to_sample_mapping = tokenized_output["overflow_to_sample_mapping"]
        text = examples[input_config.get("dataset").get("text_column")]
        text = [
            text[i] for i in overflow_to_sample_mapping
        ]  # Reorder text to match tokenized output

        return {
            **tokenized_output,
            "text": text,
        }

    columns_to_remove = [
        c
        for c in dataset["train"].column_names
        if c != input_config.get("dataset").get("text_column")
    ]
    tokenized_ds = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=columns_to_remove,
    )

    masker = KeywordMasking(
        tokenizer,
        input_config.get("dataset").get("text_column"),
        **input_config.get("dataset").get("keyword_masking"),
    )
    tokenized_ds = masker.mask_with_keywords(tokenized_ds)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    return tokenized_ds, tokenizer, data_collator
