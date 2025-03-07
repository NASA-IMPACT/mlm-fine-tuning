import copy
import random
from typing import Any, Dict, List, Optional, TextIO, Union

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline

def extend_position_embeddings_simple(model, tokenizer, new_max_position=1024):
    # Deep copy to avoid modifying originals
    tokenizer = copy.deepcopy(tokenizer)
    model = copy.deepcopy(model)
    
    # Update tokenizer settings
    tokenizer.model_max_length = new_max_position
    
    # Get the current configuration
    config = model.config
    original_max_position = config.max_position_embeddings
    
    print(f"Extending position embeddings from {original_max_position} to {new_max_position}")
    
    # Get the original position embeddings
    old_embeddings = model.roberta.embeddings.position_embeddings.weight.data
    
    # Create new position embeddings
    new_embeddings = torch.zeros(
        new_max_position, old_embeddings.size(1),
        dtype=old_embeddings.dtype,
        device=old_embeddings.device
    )
    
    # Copy the original embeddings for the positions that overlap
    new_embeddings[:original_max_position] = old_embeddings
    
    # For positions beyond the original, copy the last position embedding
    if new_max_position > original_max_position:
        new_embeddings[original_max_position:] = old_embeddings[-1].unsqueeze(0).expand(
            new_max_position - original_max_position, -1
        )
    # Randomly initialize the remaining position embeddings
    # if new_max_position > original_max_position:
    #     new_embeddings[original_max_position:] = torch.nn.init.normal_(
    #         torch.empty(new_max_position - original_max_position, old_embeddings.size(1),
    #                     dtype=old_embeddings.dtype, device=old_embeddings.device),
    #         mean=0.0, std=old_embeddings.std()
    #     )
    
    # Replace the embeddings
    model.roberta.embeddings.position_embeddings = torch.nn.Embedding.from_pretrained(
        new_embeddings,
        # freeze=model.roberta.embeddings.position_embeddings.weight.requires_grad
        freeze=False
    )
    
    # Update position IDs and token type IDs as before...
    if hasattr(model.roberta.embeddings, "position_ids"):
        new_position_ids = torch.arange(new_max_position).expand((1, -1))
        model.roberta.embeddings.register_buffer("position_ids", new_position_ids)
    
    if hasattr(model.roberta.embeddings, "token_type_ids"):
        new_token_type_ids = torch.zeros(1, new_max_position, dtype=torch.long)
        model.roberta.embeddings.register_buffer("token_type_ids", new_token_type_ids)
    
    # Update the config
    config.max_position_embeddings = new_max_position
    model.config = config

    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def printd(*args: Any, **kwargs: Any) -> None:
    """
    Prints the provided arguments. If the 'file' keyword argument is provided,
    it prints to the file and then to the default standard output.

    Args:
        *args: Positional arguments to be printed.
        **kwargs: Keyword arguments for the `print` function.
    """
    file: Optional[TextIO] = kwargs.get("file", None)

    # Print to the specified file if provided
    print(*args, **kwargs)

    # If 'file' is provided, remove it and print to default stdout
    if file is not None:
        del kwargs["file"]
        print(*args, **kwargs)


def mask_random_token(
    tokenized_example: Dict[str, List[int]],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[int]]:
    """
    Masks a random token in the input IDs of a tokenized example.

    Args:
        tokenized_example: Dictionary with tokenized data containing "input_ids".
        tokenizer: Tokenizer to handle token encoding and decoding.

    Returns:
        The updated example with one token masked and the original masked token string.
    """
    input_ids = tokenized_example["input_ids"]
    maskable_positions = [
        i
        for i in range(len(input_ids))
        if input_ids[i]
        not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    ]

    if maskable_positions:
        mask_index = random.choice(maskable_positions)
        original_token = input_ids[mask_index]
        input_ids[mask_index] = tokenizer.mask_token_id

        tokenized_example["masked_token_str"] = tokenizer.decode([original_token])
    else:
        input_ids.insert(
            1,
            tokenizer.mask_token_id,
        )  # Add <mask> after the [CLS] token
        tokenized_example["masked_token_str"] = tokenizer.mask_token

    tokenized_example["input_ids"] = input_ids
    return tokenized_example


def keyword_token_masking(
    tokenized_example: Dict[str, List[int]],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[int]]:
    """
    Masks a token in the input_ids_clone which has the unmasked input ids of a tokenized example.

    Args:
        tokenized_example: Dictionary with tokenized data containing "input_ids".
        tokenizer: Tokenizer to handle token encoding and decoding.

    Returns:
        The updated example with one token masked and the original masked token string.
    """
    input_ids = tokenized_example.get(
        "input_ids_clone",
        tokenized_example.get("input_ids"),
    )
    probability_matrix = tokenized_example["probability_matrix"]
    # the maskable position are thosw which have > 0 value in probablity matrix
    maskable_positions = [
        i for i in range(len(probability_matrix)) if probability_matrix[i] > 0.0
    ]

    if len(maskable_positions) > 0:
        mask_index = random.choice(maskable_positions)
        if isinstance(input_ids, torch.Tensor):
            original_token = input_ids[mask_index].item()  # converting tensor to int
        else:
            original_token = input_ids[mask_index]
        input_ids[mask_index] = tokenizer.mask_token_id
        tokenized_example["masked_token_str"] = str(tokenizer.decode([original_token]))

    else:
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()  # Convert tensor to list before inserting
        input_ids.insert(
            1,
            tokenizer.mask_token_id,
        )  # Add <mask> after the [CLS] token
        tokenized_example["masked_token_str"] = tokenizer.mask_token

    tokenized_example["input_ids"] = input_ids

    return tokenized_example


def arrange_inference_results(
    predictions: List[List[Dict]],
    targets: List[str],
    inputs: List[str],
) -> pd.DataFrame:
    """
    Arranges inference results and targets into a DataFrame.

    Args:
        predictions: List of prediction groups, each containing token scores and details.
        targets: List of original masked tokens as strings.

    Returns:
        DataFrame with sequences, targets, and top predictions.
    """
    data = []
    assert len(predictions) == len(targets) == len(inputs)
    for pred, target_str, input in zip(predictions, targets, inputs):
        top_predictions = []
        for p in pred:
            if type(p) == dict:
                top_predictions.append(
                    {
                        "score": round(float(p["score"]), 4),
                        "token_str": p["token_str"],
                        "token": p["token"],
                    },
                )
            else:
                print(p)

        row = {"input": input, "target": target_str}
        for i, top_pred in enumerate(top_predictions):
            row[f"top{i+1}"] = top_pred

        data.append(row)

    return pd.DataFrame(data)


def generate_inference(
    datasplit,
    tokenizer: PreTrainedTokenizer,
    model_save_loc: Union[str, PreTrainedModel],
    top_k: int = 3,
    n_predictions: Optional[int] = None,
    max_length: Optional[int] = 512,
    use_keyword_based_masking: bool = False,
) -> pd.DataFrame:
    """
    Generates inference results for a masked dataset using a trained model.

    Args:
        datasplit: Dataset split containing examples to mask and process.
        tokenizer: Tokenizer for handling token encoding/decoding.
        model_save_loc: Path to the saved model for inference.
        top_k: Number of top predictions to return per example.
        n_predictions: Optional limit on the number of predictions to process.

    Returns:
        DataFrame with masked inference results.
    """
    if n_predictions is not None:
        datasplit = datasplit.select(range(min(datasplit.num_rows, n_predictions)))

    if use_keyword_based_masking:
        masked_dataset = datasplit.map(
            lambda example: keyword_token_masking(example, tokenizer),
        )
    else:
        masked_dataset = datasplit.map(
            lambda example: mask_random_token(example, tokenizer),
        )
    decoded_texts = [
        tokenizer.decode(example["input_ids"], clean_up_tokenization_spaces=True)
        for example in masked_dataset
    ]

    mask_filler = pipeline(
        "fill-mask",
        model=model_save_loc,
        tokenizer=tokenizer,
        device_map="auto",
    )

    # filter decoded_texts: remove those without <mask> token
    decoded_texts = [text for text in decoded_texts if tokenizer.mask_token in text]
    masked_token_strs = [
        example.get("masked_token_str")
        for example, text in zip(masked_dataset, decoded_texts)
        if tokenizer.mask_token in text
    ]

    results = mask_filler(
        decoded_texts,
        top_k=top_k,
        tokenizer_kwargs={
            "truncation": True,
            "max_length": max_length,
            "add_special_tokens": False,
        },
    )
    return arrange_inference_results(results, masked_token_strs, decoded_texts)

def freeze_roberta_position_embeddings(model: PreTrainedModel) -> PreTrainedModel:
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    model.roberta.embeddings.position_embeddings.weight.requires_grad = True
    return model