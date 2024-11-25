from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


def get_dataset(load_dataset_config, val_split, test_split):
    ds = load_dataset(**load_dataset_config)

    if val_split is not None and test_split is not None:
        temp_split = ds.train_test_split(test_size=(val_split + test_split), seed=42)
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


def chunk_texts(examples, chunk_size=128):
    results = {}
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


def preprocess_dataset(input_config):
    dataset = get_dataset(
        input_config.get("dataset").get("hf"),
        input_config.get("dataset").get("val_split"),
        input_config.get("dataset").get("test_split"),
    )
    tokenizer = AutoTokenizer.from_pretrained(input_config.get("model").get("hf"))

    tokenized_ds = dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )
    lm_dataset = tokenized_ds.map(
        chunk_texts,
        batched=True,
        num_proc=4,
        fn_kwargs={"chunk_size": input_config.get("dataset").get("chunk_size")},
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=input_config.get("dataset").get("mlm_probability"),
    )

    _test = data_collator(lm_dataset[0])
    print(_test)
    return lm_dataset, tokenizer, data_collator
