from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)


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


def get_dataset(
    dataset_config: Dict[str, Any],
    data_src: str,
    n_rows: int,
):
    ds = DatasetDict(load_dataset(**dataset_config.get(data_src)))
    if n_rows:
        ds = DatasetDict(
            {split: ds[split].select(range(n_rows)) for split in ds.keys()},
        )
    return split_dataset(
        ds,
        dataset_config["val_split"],
        dataset_config["test_split"],
    )


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
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=input_config["dataset"]["mlm_probability"],
    )

    return lm_dataset, tokenizer, data_collator
