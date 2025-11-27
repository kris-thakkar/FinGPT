from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from fingpt.config import HardwareConfig, SentimentConfig, TrainingConfig


def load_tokenizer(base_model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def build_peft_config(training_config: TrainingConfig) -> LoraConfig:
    target_modules: Optional[List[str]] = training_config.lora_target_modules
    return LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
    )


def _bitsandbytes_config(hardware_config: HardwareConfig) -> Optional[BitsAndBytesConfig]:
    if not hardware_config.use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(
    config: SentimentConfig,
    num_labels: int,
    adapter: Optional[LoraConfig] = None,
) -> PreTrainedModel:
    quantization_config = _bitsandbytes_config(config.hardware)
    model_kwargs = {
        "num_labels": num_labels,
        "device_map": config.hardware.device_map,
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name,
        **model_kwargs,
    )
    if adapter is not None:
        model = get_peft_model(model, adapter)

    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if model.config.pad_token_id is None:
        tokenizer = load_tokenizer(config.base_model_name)
        model.config.pad_token_id = tokenizer.pad_token_id

    return model


def load_finetuned_model(config: SentimentConfig) -> PreTrainedModel:
    quantization_config = _bitsandbytes_config(config.hardware)
    model_kwargs = {
        "device_map": config.hardware.device_map,
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForSequenceClassification.from_pretrained(
        config.training.output_dir,
        **model_kwargs,
    )
    return model


__all__ = [
    "load_tokenizer",
    "build_peft_config",
    "load_model",
    "load_finetuned_model",
]
