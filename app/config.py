"""Central configuration for E-Reputation AI.

Keep environment-specific paths and model/runtime settings here instead of
hard-coding them throughout the Streamlit application.
"""

from __future__ import annotations

import os
from pathlib import Path


# Repository paths
PROJECT_ROOT = Path(
    os.getenv("EREPUTATION_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Model directories.
# Override these with environment variables when the local layout differs.
SENTIMENT_MODEL_DIR = Path(
    os.getenv(
        "SENTIMENT_MODEL_DIR",
        str(MODELS_DIR / "sentiment"),
    )
)

RECOMMENDATION_MODEL_DIR = Path(
    os.getenv(
        "RECOMMENDATION_MODEL_DIR",
        str(MODELS_DIR / "recommendation"),
    )
)

# Runtime defaults
DEFAULT_SENTIMENT_BATCH_SIZE = int(
    os.getenv("SENTIMENT_BATCH_SIZE", "32")
)

DEFAULT_SENTIMENT_MAX_LENGTH = int(
    os.getenv("SENTIMENT_MAX_LENGTH", "128")
)

DEFAULT_RECOMMENDATION_MAX_INPUT_LENGTH = int(
    os.getenv("RECOMMENDATION_MAX_INPUT_LENGTH", "512")
)

DEFAULT_RECOMMENDATION_MAX_NEW_TOKENS = int(
    os.getenv("RECOMMENDATION_MAX_NEW_TOKENS", "180")
)

DEFAULT_RECOMMENDATION_NUM_BEAMS = int(
    os.getenv("RECOMMENDATION_NUM_BEAMS", "4")
)

DEFAULT_TOP_K_REVIEWS = int(
    os.getenv("TOP_K_REVIEWS", "5")
)

# Application metadata
APP_TITLE = "E-Reputation AI"
APP_DESCRIPTION = (
    "AI-powered hotel review analysis and actionable "
    "e-reputation recommendations."
)


def ensure_project_directories() -> None:
    """Create local runtime directories when they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
