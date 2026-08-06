"""Business analytics and prioritization for E-Reputation AI.

This module contains presentation-independent calculations used by the
dashboard: aspect ranking, impact scores, priority levels, and Pareto data.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


PRIORITY_THRESHOLDS = {
    "Critical": 30.0,
    "High": 18.0,
    "Medium": 8.0,
}


def calculate_impact_score(
    mentions: float,
    negative_review_count: int | float,
) -> float:
    """Calculate the impact score from aspect mentions.

    Impact is expressed as the percentage of negative reviews associated
    with the aspect.
    """
    denominator = max(float(negative_review_count), 1.0)
    return float(mentions) / denominator * 100.0


def classify_priority(impact_score: float) -> str:
    """Map an impact score to the project's business priority levels."""
    score = float(impact_score)

    if score >= PRIORITY_THRESHOLDS["Critical"]:
        return "Critical"
    if score >= PRIORITY_THRESHOLDS["High"]:
        return "High"
    if score >= PRIORITY_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


def build_business_aspect_table(
    aspect_summary: pd.DataFrame,
    negative_review_count: int | float,
) -> pd.DataFrame:
    """Build the business-facing aspect ranking table."""
    if aspect_summary is None or aspect_summary.empty:
        return pd.DataFrame(
            columns=[
                "aspect",
                "mentions",
                "avg_confidence",
                "percentage",
                "impact_score",
                "priority",
            ]
        )

    required = {"aspect", "mentions"}
    missing = required - set(aspect_summary.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    result = aspect_summary.copy()

    result["mentions"] = pd.to_numeric(
        result["mentions"], errors="coerce"
    ).fillna(0)

    result["impact_score"] = result["mentions"].apply(
        lambda mentions: calculate_impact_score(
            mentions,
            negative_review_count,
        )
    )

    result["priority"] = result["impact_score"].apply(classify_priority)

    result = result.sort_values(
        ["impact_score", "mentions"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result


def build_pareto_data(
    business_aspects: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare aspect mentions and cumulative percentage for Pareto charts."""
    if business_aspects is None or business_aspects.empty:
        return pd.DataFrame(
            columns=["aspect", "mentions", "cumulative_percentage"]
        )

    result = business_aspects[["aspect", "mentions"]].copy()
    result["mentions"] = pd.to_numeric(
        result["mentions"], errors="coerce"
    ).fillna(0)

    total = result["mentions"].sum()

    if total <= 0:
        result["cumulative_percentage"] = 0.0
        return result

    result = result.sort_values("mentions", ascending=False).reset_index(drop=True)
    result["cumulative_percentage"] = (
        result["mentions"].cumsum() / total * 100.0
    )

    return result


def build_priority_matrix_data(
    business_aspects: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare aspect-level data for an impact/confidence priority matrix."""
    if business_aspects is None or business_aspects.empty:
        return pd.DataFrame(
            columns=["aspect", "impact_score", "avg_confidence", "priority"]
        )

    columns = ["aspect", "impact_score", "priority"]

    if "avg_confidence" in business_aspects.columns:
        columns.insert(2, "avg_confidence")

    return business_aspects[columns].copy()


def summarize_priority_counts(
    business_aspects: pd.DataFrame,
) -> pd.DataFrame:
    """Return the number of aspects in each priority category."""
    priorities = ["Critical", "High", "Medium", "Low"]

    if business_aspects is None or business_aspects.empty:
        return pd.DataFrame(
            {"priority": priorities, "count": [0] * len(priorities)}
        )

    counts = (
        business_aspects["priority"]
        .value_counts()
        .reindex(priorities, fill_value=0)
        .rename_axis("priority")
        .reset_index(name="count")
    )

    return counts
