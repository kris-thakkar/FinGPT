from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    train_file: str
    eval_file: str
    text_column: str
    label_column: str
    max_length: int = 256


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 2
    lr_scheduler_type: str = "linear"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    gradient_checkpointing: bool = False
    use_bf16: bool = False
    seed: int = 42


@dataclass
class HardwareConfig:
    use_4bit: bool = False
    device_map: Optional[str] = "auto"


@dataclass
class SentimentConfig:
    base_model_name: str
    labels: List[str]
    data: DataConfig
    training: TrainingConfig
    hardware: HardwareConfig


def _build_config_dict(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(config_path: str) -> SentimentConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_config = _build_config_dict(path)
    data_cfg = DataConfig(**raw_config["data"])
    training_cfg = TrainingConfig(**raw_config["training"])
    hardware_cfg = HardwareConfig(**raw_config.get("hardware", {}))
    return SentimentConfig(
        base_model_name=raw_config["base_model_name"],
        labels=raw_config["labels"],
        data=data_cfg,
        training=training_cfg,
        hardware=hardware_cfg,
    )


__all__ = [
    "DataConfig",
    "TrainingConfig",
    "HardwareConfig",
    "SentimentConfig",
    "load_config",
]
