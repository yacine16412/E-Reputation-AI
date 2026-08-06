# Model Documentation

## Overview

E-Reputation AI uses fine-tuned and pretrained NLP components for sentiment
analysis, recommendation generation, semantic similarity, and rule-based
aspect detection.

## 1. Sentiment Analysis — DistilBERT

**Model:** DistilBERT fine-tuned for binary sentiment classification.

**Purpose:** Classifies customer reviews as:

- Positive
- Negative

The fine-tuned weights are not committed to GitHub because the local model
directory is approximately 8 GB.

### Reproduction

The fine-tuning process is documented in the notebooks included in the
repository. A clean environment should be installed from `requirements.txt`
before reproducing the training process.

## 2. Recommendation Generation — FLAN-T5

**Model:** FLAN-T5 fine-tuned for recommendation generation.

**Purpose:** Uses representative negative customer comments and detected
business aspects to generate:

- issue summaries
- actionable recommendations

The fine-tuned weights are not committed to GitHub.

## 3. Semantic Similarity — Sentence Transformers

**Model:** `all-MiniLM-L6-v2`.

**Purpose:** Provides semantic representations used for similarity-based
processing and representative review selection where applicable.

## 4. Rule-Based Components

The system also uses:

- Regex
- keyword matching

These components support text preprocessing and aspect detection.

## Model Storage Policy

Fine-tuned model weights are intentionally excluded from GitHub because of
their size.

See `.gitignore` and the repository's model directory documentation.

For reproducibility, use the training notebooks rather than committing the
large model artifacts.
