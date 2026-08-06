"""Text preprocessing utilities for E-Reputation AI.

This module centralizes review-text normalization used by the sentiment
analysis and rule-based aspect extraction pipelines.
"""

from __future__ import annotations

import re
import unicodedata


EMOJI_TOKEN_MAP = {
    "😀": " positive_emoji ",
    "😁": " positive_emoji ",
    "😂": " positive_emoji ",
    "😊": " positive_emoji ",
    "😍": " positive_emoji ",
    "🤩": " positive_emoji ",
    "👍": " positive_emoji ",
    "🙏": " positive_emoji ",
    "❤": " positive_emoji ",
    "❤️": " positive_emoji ",
    "😃": " positive_emoji ",
    "😄": " positive_emoji ",
    "🙂": " positive_emoji ",
    "😞": " negative_emoji ",
    "😠": " negative_emoji ",
    "😡": " negative_emoji ",
    "😢": " negative_emoji ",
    "👎": " negative_emoji ",
    "💔": " negative_emoji ",
    "😕": " negative_emoji ",
    "🙁": " negative_emoji ",
}

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"\s+")
NON_LETTER_RE = re.compile(r"[^a-zA-Z\s']+")


def strip_accents(text: str) -> str:
    """Remove accents while preserving readable Latin text."""
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def replace_emojis(text: str) -> str:
    """Preserve sentiment-bearing emojis as text tokens and remove others."""
    if not text:
        return ""

    text = str(text)
    for emoji_char, token in EMOJI_TOKEN_MAP.items():
        text = text.replace(emoji_char, token)

    return EMOJI_RE.sub(" ", text)


def clean_comment(text: str) -> str:
    """Clean review text for model inference and aspect matching."""
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = replace_emojis(text)
    text = strip_accents(text)
    text = text.replace("&", " and ")
    text = NON_LETTER_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    return text


def normalize_for_matching(text: str) -> str:
    """Normalize review text before keyword-based aspect extraction."""
    text = clean_comment(text)

    replacements = {
        "wi fi": "wifi",
        "wlan": "wifi",
        "check in": "checkin",
        "check out": "checkout",
        "air condition": "air conditioning",
        "a c": "air conditioning",
        "ac ": "air conditioning ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return MULTI_SPACE_RE.sub(" ", text).strip()
