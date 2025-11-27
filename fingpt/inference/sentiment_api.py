import os
from typing import Dict

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fingpt.config import load_config
from fingpt.models.sentiment_model import load_finetuned_model, load_tokenizer

DEFAULT_CONFIG_PATH = "config/sentiment_config.yaml"


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    probabilities: Dict[str, float]


def load_resources(config_path: str):
    config = load_config(config_path)
    tokenizer = load_tokenizer(config.training.output_dir)
    model = load_finetuned_model(config)
    model.eval()
    return config, tokenizer, model


def create_app(config_path: str = DEFAULT_CONFIG_PATH) -> FastAPI:
    config, tokenizer, model = load_resources(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not getattr(model, "hf_device_map", None):
        model.to(device)

    app = FastAPI(title="FinGPT Sentiment API", version="1.0")

    @app.post("/predict", response_model=SentimentResponse)
    def predict(request: SentimentRequest) -> SentimentResponse:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        encoded = tokenizer(
            request.text,
            padding="max_length",
            truncation=True,
            max_length=config.data.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1).cpu().squeeze(0)

        best_label_idx = int(torch.argmax(probabilities).item())
        label = config.labels[best_label_idx]
        prob_dict = {config.labels[i]: float(probabilities[i].item()) for i in range(len(config.labels))}

        return SentimentResponse(label=label, probabilities=prob_dict)

    return app


def get_app() -> FastAPI:
    config_path = os.environ.get("FINGPT_CONFIG", DEFAULT_CONFIG_PATH)
    return create_app(config_path)


app = get_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fingpt.inference.sentiment_api:app", host="0.0.0.0", port=8000, reload=False)
