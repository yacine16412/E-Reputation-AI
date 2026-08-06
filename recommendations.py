"""Recommendation generation service for E-Reputation AI.

This module contains the evidence-selection and FLAN-T5 generation logic.
The Streamlit UI should call these functions rather than implementing
recommendation generation itself.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MAX_INPUT_LENGTH = 512
DEFAULT_MAX_NEW_TOKENS = 180
DEFAULT_NUM_BEAMS = 4
DEFAULT_TOP_K_REVIEWS = 5


def select_representative_reviews(
    aspect_df: pd.DataFrame,
    aspect: str,
    *,
    top_k: int = DEFAULT_TOP_K_REVIEWS,
    min_words: int = 5,
) -> list[str]:
    """Select diverse, high-confidence review evidence for one aspect.

    The function expects an ``aspect_df`` containing ``aspect``,
    ``clean_text`` and, when available, ``confidence`` columns.
    """
    if aspect_df is None or aspect_df.empty:
        return []

    required = {"aspect", "clean_text"}
    missing = required - set(aspect_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    subset = aspect_df[
        aspect_df["aspect"].astype(str).str.upper() == str(aspect).upper()
    ].copy()

    if subset.empty:
        return []

    subset["clean_text"] = subset["clean_text"].fillna("").astype(str).str.strip()
    subset = subset[subset["clean_text"].str.split().str.len() >= min_words]

    if subset.empty:
        return []

    if "confidence" not in subset.columns:
        subset["confidence"] = 0.0

    subset["confidence"] = pd.to_numeric(
        subset["confidence"], errors="coerce"
    ).fillna(0.0)

    subset["word_count"] = subset["clean_text"].str.split().str.len()

    # Prefer stronger model confidence and sufficiently informative reviews.
    subset = subset.sort_values(
        ["confidence", "word_count"],
        ascending=[False, False],
    )

    # Remove exact duplicates while preserving the ranking.
    subset = subset.drop_duplicates(subset=["clean_text"])

    return subset["clean_text"].head(top_k).tolist()


def load_recommendation_model(
    model_dir: str | Path,
    device: str | torch.device | None = None,
):
    """Load a local fine-tuned FLAN-T5 tokenizer and model."""
    model_dir = Path(model_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def build_recommendation_prompt(
    aspect: str,
    representative_reviews: list[str],
) -> str:
    """Build the structured prompt used by the recommendation model."""
    evidence = "
".join(
        f"- {review}" for review in representative_reviews
    )

    return f"""You are a hotel reputation analyst.

Aspect: {aspect}

Representative negative customer reviews:
{evidence}

Analyze the customer complaints and produce a concise business-oriented response.

Issue summary: <summarize the main problem>
Recommendation: <give practical actions the hotel can take>

Do not invent facts that are not supported by the reviews.
"""


def parse_recommendation_output(text: str) -> tuple[str, str]:
    """Extract issue summary and recommendation from generated text."""
    text = str(text or "").strip()

    issue_match = re.search(
        r"Issue\s*summary\s*:\s*(.*?)(?=
\s*Recommendation\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    recommendation_match = re.search(
        r"Recommendation\s*:\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    issue = (
        issue_match.group(1).strip()
        if issue_match
        else ""
    )
    recommendation = (
        recommendation_match.group(1).strip()
        if recommendation_match
        else ""
    )

    if not issue and not recommendation:
        return text, ""

    return issue, recommendation


def generate_recommendation(
    aspect: str,
    representative_reviews: list[str],
    tokenizer,
    model,
    device: str | torch.device,
    *,
    max_input_length: int = DEFAULT_MAX_INPUT_LENGTH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = DEFAULT_NUM_BEAMS,
) -> tuple[str, str]:
    """Generate an issue summary and actionable recommendation."""
    if not representative_reviews:
        return (
            "No representative negative reviews were available.",
            "No recommendation can be generated for this aspect.",
        )

    prompt = build_recommendation_prompt(
        aspect,
        representative_reviews,
    )

    resolved_device = torch.device(device)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )

    encoded = {
        key: value.to(resolved_device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()

    return parse_recommendation_output(generated_text)
