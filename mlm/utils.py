import random
from typing import Any, List, Optional, TextIO, Dict
from transformers import PreTrainedTokenizer, pipeline
import pandas as pd

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
    tokenizer: PreTrainedTokenizer
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
        i for i in range(len(input_ids))
        if input_ids[i] not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    ]

    if maskable_positions:
        mask_index = random.choice(maskable_positions)
        original_token = input_ids[mask_index]
        input_ids[mask_index] = tokenizer.mask_token_id

        tokenized_example["masked_token_str"] = tokenizer.decode([original_token])
    
    tokenized_example["input_ids"] = input_ids
    return tokenized_example


def arrange_inference_results(
    predictions: List[List[Dict]], 
    targets: List[str]
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
    for pred, target_str in zip(predictions, targets):
        sequence = pred[0]["sequence"]
        top_predictions = [
            {
                "score": round(p["score"], 4),
                "token_str": p["token_str"],
                "token": p["token"],
            }
            for p in pred
        ]

        row = {"sequence": sequence, "target": target_str}
        for i, top_pred in enumerate(top_predictions):
            row[f"top{i+1}"] = top_pred

        data.append(row)
    
    return pd.DataFrame(data)


def generate_inference(
    datasplit, 
    tokenizer: PreTrainedTokenizer, 
    model_save_loc: str, 
    top_k: int = 3, 
    n_predictions: Optional[int] = None
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
    
    masked_dataset = datasplit.map(lambda example: mask_random_token(example, tokenizer))
    decoded_texts = [tokenizer.decode(example["input_ids"]) for example in masked_dataset]
    
    mask_filler = pipeline("fill-mask", model=model_save_loc, tokenizer=model_save_loc)
    results = mask_filler(decoded_texts, top_k=top_k)
    
    return arrange_inference_results(results, masked_dataset["masked_token_str"])