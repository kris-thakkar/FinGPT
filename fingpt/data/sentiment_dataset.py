from typing import Dict, List

import datasets
from transformers import PreTrainedTokenizerBase

from fingpt.config import DataConfig


def load_sentiment_datasets(data_config: DataConfig) -> Dict[str, datasets.Dataset]:
    data_files = {
        "train": data_config.train_file,
        "validation": data_config.eval_file,
    }
    dataset_dict = datasets.load_dataset("parquet", data_files=data_files)
    return {
        "train": dataset_dict["train"],
        "validation": dataset_dict["validation"],
    }


def filter_by_labels(
    datasets_dict: Dict[str, datasets.Dataset],
    data_config: DataConfig,
    allowed_labels: List[str],
) -> Dict[str, datasets.Dataset]:
    def keep_example(example: dict) -> bool:
        return example[data_config.label_column] in allowed_labels

    filtered = {}
    for split_name, ds in datasets_dict.items():
        filtered[split_name] = ds.filter(keep_example)
    return filtered


def build_label_to_id(labels: List[str]) -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(labels)}


def tokenize_function(
    example: Dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    data_config: DataConfig,
    label_to_id: Dict[str, int],
) -> Dict[str, List[int]]:
    text = example[data_config.text_column]
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=data_config.max_length,
    )
    tokenized["labels"] = label_to_id.get(example[data_config.label_column], -1)
    return tokenized


def tokenize_datasets(
    datasets_dict: Dict[str, datasets.Dataset],
    tokenizer: PreTrainedTokenizerBase,
    data_config: DataConfig,
    labels: List[str],
) -> Dict[str, datasets.Dataset]:
    label_to_id = build_label_to_id(labels)
    tokenization_fn = lambda example: tokenize_function(
        example, tokenizer=tokenizer, data_config=data_config, label_to_id=label_to_id
    )

    tokenized = {}
    for split_name, ds in datasets_dict.items():
        mapped = ds.map(tokenization_fn, batched=True, remove_columns=ds.column_names)
        tokenized[split_name] = mapped
    return tokenized
