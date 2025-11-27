# FinGPT Quickstart

Follow these steps to go from a clean checkout to a running sentiment model and API.

## 1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If `bitsandbytes` fails to install on your platform, comment it out in `requirements.txt` and set `hardware.use_4bit: false` in `config/sentiment_config.yaml`.

## 2) Prepare the labeled dataset

A starter CSV is provided at `data/raw/fin_sentiment_labeled.csv` with two columns: `text` and `label`. You can edit or extend it with your own examples as long as labels match those listed under `labels` in `config/sentiment_config.yaml`.

Convert the CSV into train/eval parquet files:

```bash
python scripts/prepare_sentiment_data.py --input-csv data/raw/fin_sentiment_labeled.csv
```

This writes `data/processed/sentiment_train.parquet` and `data/processed/sentiment_eval.parquet`.

## 3) Fine-tune the sentiment model with LoRA

```bash
python -m fingpt.training.train_sentiment_lora --config config/sentiment_config.yaml
```

The script loads the base model from the config, applies LoRA adapters, trains with accuracy and macro-F1, and saves a checkpoint to `checkpoints/fingpt-sentiment-v1` by default.

## 4) Serve the inference API

After training finishes, launch the FastAPI app:

```bash
uvicorn fingpt.inference.sentiment_api:app --host 0.0.0.0 --port 8000
```

Send a sample request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Earnings beat expectations and guidance was raised."}'
```

You will receive a JSON response with the predicted label and per-class probabilities.

## 5) Adjust configuration as needed

- Update `base_model_name` to choose a different Hugging Face model.
- Tweak `training` hyperparameters (epochs, batch sizes, learning rate, LoRA settings).
- Toggle `hardware.use_4bit` for 4-bit quantization (requires `bitsandbytes`).

## 6) (Optional) Override config for API runs

You can point the API to a custom config by setting an environment variable before starting Uvicorn:

```bash
FINGPT_CONFIG=/path/to/custom.yaml uvicorn fingpt.inference.sentiment_api:app --host 0.0.0.0 --port 8000
```

This is useful if your model checkpoint or label set differs from the defaults.
