import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments, set_seed

from fingpt.config import SentimentConfig, load_config
from fingpt.data.sentiment_dataset import (
    build_label_to_id,
    filter_by_labels,
    load_sentiment_datasets,
    tokenize_datasets,
)
from fingpt.models.sentiment_model import build_peft_config, load_model, load_tokenizer


def compute_metrics_builder(id_to_label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average="macro")
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

    return compute_metrics


def prepare_datasets(config: SentimentConfig, tokenizer):
    datasets_dict = load_sentiment_datasets(config.data)
    datasets_dict = filter_by_labels(datasets_dict, config.data, config.labels)
    tokenized = tokenize_datasets(datasets_dict, tokenizer, config.data, config.labels)

    label_columns = ["input_ids", "attention_mask", "labels"]
    for split_name, ds in tokenized.items():
        tokenized[split_name] = ds.with_format("torch", columns=label_columns)
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA-tuned sentiment classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sentiment_config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.training.seed)

    tokenizer = load_tokenizer(config.base_model_name)
    label_to_id = build_label_to_id(config.labels)
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    tokenized = prepare_datasets(config, tokenizer)

    peft_config = build_peft_config(config.training)
    model = load_model(config, num_labels=len(config.labels), adapter=peft_config)
    model.config.label2id = label_to_id
    model.config.id2label = id_to_label

    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        lr_scheduler_type=config.training.lr_scheduler_type,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        bf16=config.training.use_bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_builder(id_to_label),
    )

    trainer.train()

    save_path = Path(config.training.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
