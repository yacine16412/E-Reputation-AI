"""Rule-based aspect extraction for E-Reputation AI.

The module detects business-relevant aspects in hotel reviews using
normalized keyword matching and regular expressions.
"""

from __future__ import annotations

import re
from typing import Iterable

from preprocessing import normalize_for_matching


ASPECT_KEYWORDS: dict[str, list[str]] = {
    "ROOM": [
        "room", "bed", "bedroom", "bathroom", "shower", "toilet",
        "mattress", "pillow", "towel", "balcony", "air conditioning",
        "air conditioner", "ac", "heating", "wardrobe", "closet",
    ],
    "STAFF": [
        "staff", "reception", "receptionist", "service", "manager",
        "employee", "worker", "host", "waiter", "waitress",
    ],
    "FOOD": [
        "food", "breakfast", "restaurant", "dinner", "lunch", "meal",
        "buffet", "coffee", "drink", "bar", "menu",
    ],
    "WIFI": [
        "wifi", "wi-fi", "internet", "connection", "network",
        "wireless", "signal",
    ],
    "CLEANLINESS": [
        "clean", "cleanliness", "dirty", "dirt", "dust", "smell",
        "odor", "hygiene", "stain", "filthy", "unclean",
    ],
    "NOISE": [
        "noise", "noisy", "loud", "sound", "quiet", "traffic",
        "music", "party", "disturbance",
    ],
    "LOCATION": [
        "location", "area", "place", "beach", "city center", "centre",
        "downtown", "airport", "station", "near", "far", "distance",
    ],
    "VALUE": [
        "price", "value", "expensive", "cheap", "cost", "money",
        "worth", "overpriced", "refund",
    ],
    "FACILITIES": [
        "facility", "facilities", "pool", "swimming pool", "gym",
        "spa", "parking", "elevator", "lift", "garden", "terrace",
        "equipment",
    ],
}


def build_aspect_patterns(
    aspect_keywords: dict[str, Iterable[str]] = ASPECT_KEYWORDS,
) -> dict[str, re.Pattern[str]]:
    """Compile case-insensitive regex patterns for each aspect."""
    patterns: dict[str, re.Pattern[str]] = {}

    for aspect, keywords in aspect_keywords.items():
        escaped_keywords = sorted(
            {re.escape(str(keyword).strip()) for keyword in keywords if str(keyword).strip()},
            key=len,
            reverse=True,
        )
        if not escaped_keywords:
            continue

        patterns[aspect] = re.compile(
            r"\b(?:"
            + "|".join(escaped_keywords)
            + r")\b",
            flags=re.IGNORECASE,
        )

    return patterns


ASPECT_PATTERNS = build_aspect_patterns()


def extract_aspects(
    text: str,
    patterns: dict[str, re.Pattern[str]] = ASPECT_PATTERNS,
) -> list[str]:
    """Return all aspects detected in a review."""
    normalized_text = normalize_for_matching(text)

    if not normalized_text:
        return []

    return [
        aspect
        for aspect, pattern in patterns.items()
        if pattern.search(normalized_text)
    ]


def build_aspect_analysis(
    negative_reviews,
    patterns: dict[str, re.Pattern[str]] = ASPECT_PATTERNS,
):
    """Add detected aspects and return aspect-level summary statistics.

    The input is expected to be a pandas DataFrame containing a
    ``clean_text`` column. The function intentionally keeps pandas local
    to this analytics boundary so the extraction logic remains reusable.
    """
    import pandas as pd

    if negative_reviews is None:
        negative_reviews = pd.DataFrame()

    df = negative_reviews.copy()

    if "clean_text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'clean_text' column.")

    df["aspects"] = df["clean_text"].apply(
        lambda text: extract_aspects(text, patterns)
    )

    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        for aspect in row["aspects"]:
            rows.append(
                {
                    "aspect": aspect,
                    "confidence": row.get("confidence"),
                    "clean_text": row.get("clean_text", ""),
                }
            )

    if not rows:
        return df, pd.DataFrame(
            columns=["aspect", "mentions", "avg_confidence", "percentage"]
        )

    aspect_df = pd.DataFrame(rows)

    summary = (
        aspect_df.groupby("aspect")
        .agg(
            mentions=("aspect", "size"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    total_negative_reviews = max(len(df), 1)
    summary["percentage"] = (
        summary["mentions"] / total_negative_reviews * 100
    )

    summary = summary.sort_values(
        ["mentions", "avg_confidence"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return df, summary
