import argparse
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from fingpt.config import load_config


def validate_labels(df: pd.DataFrame, label_column: str, allowed_labels: List[str]) -> pd.DataFrame:
    filtered = df[df[label_column].isin(allowed_labels)].copy()
    return filtered


def prepare_dataset(
    input_csv: Path,
    train_output: Path,
    eval_output: Path,
    config_path: Path,
    test_size: float = 0.2,
    seed: int = 42,
):
    config = load_config(str(config_path))

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=[config.data.text_column, config.data.label_column])
    df = validate_labels(df, config.data.label_column, config.labels)

    train_df, eval_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[config.data.label_column],
        shuffle=True,
    )

    train_output.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(train_output, index=False)
    eval_df.to_parquet(eval_output, index=False)

    print(f"Saved train set to {train_output} with {len(train_df)} rows")
    print(f"Saved eval set to {eval_output} with {len(eval_df)} rows")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare sentiment training data")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/raw/fin_sentiment_labeled.csv"),
        help="Path to the labeled CSV file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/sentiment_config.yaml"),
        help="Path to the sentiment YAML config.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for evaluation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/eval split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(str(args.config))
    prepare_dataset(
        input_csv=args.input_csv,
        train_output=Path(config.data.train_file),
        eval_output=Path(config.data.eval_file),
        config_path=Path(args.config),
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
