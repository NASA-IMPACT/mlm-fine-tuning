from typing import Any, Dict, List, Optional, TextIO

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForMaskedLM
from utils import printd


def get_model(
    config: Dict[str, Any],
    train_techs: List[str],
    file: Optional[TextIO] = None,
) -> AutoModelForMaskedLM:
    """
    Returns a pre-trained model based on the provided configuration and training techniques.

    Args:
    - config (Dict[str, Any]): A dictionary containing the model configuration.
    - train_techs (List[str]): A list of training techniques (e.g., 'quant', 'lora').

    Returns:
    - AutoModelForMaskedLM: The model instance ready for training or inference.
    """

    # Check if quantization is required
    if "quant" in {tech.lower() for tech in train_techs}:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            **config.get("quant", {}).get("bnb_config", {}),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # Load model with quantization configuration
        model = AutoModelForMaskedLM.from_pretrained(
            config.get("input", {}).get("model", {}).get("hf", ""),
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        printd("Using Quantization", file=file)
    else:
        # Load the model without quantization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = AutoModelForMaskedLM.from_pretrained(
                config.get("input", {}).get("model", {}).get("hf", ""),
                torch_dtype="auto",  # Uses BF16/FP16 if available
                attn_implementation="flash_attention_2",  # Enable FlashAttention
                reference_compile=False,
            ).to(device)
        except Exception as e:
            model = AutoModelForMaskedLM.from_pretrained(
                config.get("input", {}).get("model", {}).get("hf", ""),
                torch_dtype="auto",  # Uses BF16/FP16 if available
            ).to(device)

    # Check if LoRA (Low-Rank Adaptation) is required
    if "lora" in {tech.lower() for tech in train_techs}:
        peft_config = LoraConfig(**config.get("config", {}).get("peft", {}))
        # Apply LoRA model configuration
        model = get_peft_model(model, peft_config)
        printd("Using LORA", file=file)

    return model
