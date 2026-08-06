"""Sentiment inference service for E-Reputation AI.

This module isolates DistilBERT model loading and batch inference from
the Streamlit presentation layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 32


def resolve_device(preferred: str | None = None) -> torch.device:
    """Resolve the inference device, preferring CUDA when available."""
    if preferred:
        return torch.device(preferred)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sentiment_model(
    model_dir: str | Path,
    device: str | torch.device | None = None,
):
    """Load a fine-tuned DistilBERT tokenizer and classifier."""
    model_dir = Path(model_dir)
    resolved_device = (
        torch.device(device)
        if device is not None
        else resolve_device()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    model.to(resolved_device)
    model.eval()

    return tokenizer, model, resolved_device


def normalize_label(
    predicted_id: int,
    id2label: dict[int | str, str] | None = None,
) -> str:
    """Normalize a model class label to Positive or Negative."""
    if id2label:
        label = id2label.get(predicted_id, id2label.get(str(predicted_id), ""))
        label = str(label).strip().lower()

        if "positive" in label:
            return "Positive"
        if "negative" in label:
            return "Negative"

    # Fallback for the project's binary sentiment convention.
    return "Positive" if int(predicted_id) == 1 else "Negative"


def predict_sentiment_batch(
    texts: list[str],
    tokenizer,
    model,
    device: str | torch.device,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> tuple[list[str], list[float]]:
    """Predict sentiment labels and confidence scores for a list of texts."""
    if not texts:
        return [], []

    resolved_device = torch.device(device)
    labels: list[str] = []
    confidences: list[float] = []

    id2label = getattr(model.config, "id2label", None)

    for start in range(0, len(texts), batch_size):
        batch_texts = [
            "" if text is None else str(text)
            for text in texts[start : start + batch_size]
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(resolved_device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = model(**encoded)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_ids = torch.argmax(probabilities, dim=-1)

        for probability_row, predicted_id in zip(
            probabilities.detach().cpu().numpy(),
            predicted_ids.detach().cpu().numpy(),
        ):
            predicted_id = int(predicted_id)
            labels.append(normalize_label(predicted_id, id2label))
            confidences.append(float(np.max(probability_row)))

    return labels, confidences
