# FinGPT

A minimal "FinGPT"-style stack for fine-tuning an open-source LLM on financial sentiment data using LoRA, then serving real-time predictions with FastAPI.

## Project layout

- `config/sentiment_config.yaml` — training and inference configuration.
- `data/raw/` — place the labeled CSV (`fin_sentiment_labeled.csv`).
- `data/processed/` — train/eval parquet files produced by the prep script.
- `checkpoints/` — saved fine-tuned model artifacts.
- `scripts/prepare_sentiment_data.py` — split labeled CSV into train/eval parquet files.
- `fingpt/`
  - `config.py` — structured config loader.
  - `data/sentiment_dataset.py` — dataset loading, filtering, and tokenization utilities.
  - `models/sentiment_model.py` — tokenizer/model helpers plus LoRA and quantization wiring.
  - `training/train_sentiment_lora.py` — training entrypoint with HF `Trainer`.
  - `inference/sentiment_api.py` — FastAPI app exposing `POST /predict`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Tip: if `bitsandbytes` fails to install, comment it out in `requirements.txt` and set `hardware.use_4bit: false` in the config.

## Data preparation

Place a labeled CSV at `data/raw/fin_sentiment_labeled.csv` with columns `text` and `label` (labels should match the `labels` list in the config). Then run:

```bash
python scripts/prepare_sentiment_data.py --input-csv data/raw/fin_sentiment_labeled.csv
```

This writes `data/processed/sentiment_train.parquet` and `data/processed/sentiment_eval.parquet`.

## Training

```bash
python -m fingpt.training.train_sentiment_lora --config config/sentiment_config.yaml
```

The script loads the base model defined in the config, applies LoRA adapters, trains with accuracy and macro-F1 metrics, and saves the fine-tuned checkpoint to `checkpoints/fingpt-sentiment-v1`.

## Inference API

Start the FastAPI server (after training):

```bash
uvicorn fingpt.inference.sentiment_api:app --host 0.0.0.0 --port 8000
```

Send a request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Earnings beat expectations and guidance was raised."}'
```

Response shape:

```json
{
  "label": "bullish",
  "probabilities": {
    "very_bearish": 0.01,
    "bearish": 0.05,
    "neutral": 0.12,
    "bullish": 0.62,
    "very_bullish": 0.20
  }
}
```

You can override the config path for the API by setting `FINGPT_CONFIG=/path/to/config.yaml` before launching uvicorn.

## Configuration highlights

- Adjust `base_model_name` to point to your chosen Hugging Face model (e.g., Llama, Mistral).
- Edit `training` fields to tune epochs, batch sizes, learning rate, and LoRA hyperparameters.
- Set `hardware.use_4bit` to `true` to enable 4-bit quantization (requires `bitsandbytes`).
