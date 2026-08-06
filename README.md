# E-Reputation AI

> AI-Powered Online Reputation Management Platform for analyzing customer reviews, identifying business issues, and generating actionable recommendations.

## Overview

**E-Reputation AI** is an AI/NLP platform designed to help hospitality decision-makers understand online customer feedback.

The system processes hotel review datasets and provides:

- sentiment classification
- negative-review analysis
- aspect detection
- representative comment selection
- priority ranking
- AI-generated recommendations
- dashboards and KPIs
- CSV export and reporting

The project was developed as a Master's graduation project and is being prepared as a professional open-source AI repository.

## Core Pipeline

```text
Customer Reviews
       ↓
CSV Upload
       ↓
Text Preprocessing
       ↓
DistilBERT
       ↓
Sentiment Classification
       ↓
Negative Reviews
       ↓
Aspect Detection
       ↓
Representative Comments
       ↓
FLAN-T5
       ↓
Issue Summary + Recommendation
       ↓
Analytics / Dashboard
       ↓
Export
```

## Key Features

### Review Analysis
- CSV upload
- configurable review-column selection
- configurable number of reviews
- configurable batch size
- text cleaning and normalization

### AI Analysis
- binary sentiment classification
- negative-review filtering
- aspect/problem detection
- representative review selection
- recommendation generation

### Decision Support
- KPI cards
- statistics
- priority ranking
- Pareto analysis
- actionable recommendations
- downloadable CSV results

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | Streamlit, HTML, CSS, Bootstrap Icons |
| Backend | Python, Streamlit |
| Data | Pandas, CSV |
| Sentiment | DistilBERT |
| Generation | FLAN-T5 |
| Semantic processing | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Rule-based NLP | Regex, keyword matching |

## Project Structure

```text
E-Reputation-AI/
├── app/
│   ├── app.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── aspects.py
│   ├── recommendations.py
│   ├── analytics.py
│   └── export.py
├── data/
├── models/
├── notebooks/
├── outputs/
├── docs/
├── screenshots/
├── MODELS.md
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd E-Reputation-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare models

The fine-tuned model weights are intentionally not included in the repository because of their size.

See [`MODELS.md`](MODELS.md) for the model policy and reproduction information.

### 5. Run the application

```bash
streamlit run app/app.py
```

## Usage

1. Start the Streamlit application.
2. Upload a hotel-review CSV file.
3. Select the column containing customer reviews.
4. Select the maximum number of reviews to analyze.
5. Select the batch size.
6. Run the analysis.
7. Review sentiment results.
8. Inspect negative aspects and business priorities.
9. Review generated recommendations.
10. Export the results.

## Models

### DistilBERT

Used for binary sentiment classification:

```text
Positive
Negative
```

### FLAN-T5

Used to transform representative negative-review evidence into:

- issue summaries
- actionable recommendations

### Sentence Transformers

`all-MiniLM-L6-v2` is used for semantic representation and similarity-based processing where applicable.

### Rule-Based NLP

Regex and keyword matching support preprocessing and aspect detection.

For details, see [`MODELS.md`](MODELS.md).

## Datasets

The project uses:

1. Booking.com Reviews Dataset
2. A custom recommendation dataset containing **3,965 examples**

Dataset files are not automatically assumed to be redistributable. Check the corresponding dataset license/terms before publishing any raw data.

## Reproducibility

The repository keeps the training notebooks and documentation while excluding large fine-tuned model artifacts.

The intended workflow is:

```text
Dataset
   ↓
Training Notebook
   ↓
Fine-tuning
   ↓
Evaluation
   ↓
Saved Model
   ↓
Application
```

## Project Status

**Status: Functional prototype / research project**

The application supports the complete intended analysis workflow, while the repository is being cleaned and documented for reproducibility and open-source publication.

## Academic Context

This project was developed as part of a Master's graduation project focused on AI-assisted hotel e-reputation analysis.

It combines:

- Natural Language Processing
- Transformer models
- sentiment analysis
- aspect/problem detection
- text generation
- business intelligence
- decision support

## Limitations

- Fine-tuned model weights are not stored in GitHub.
- The initial prototype primarily targets English-language reviews.
- Raw datasets may have external licensing restrictions.
- Runtime performance depends on available CPU/GPU resources.

## Future Improvements

Potential extensions include:

- multilingual sentiment analysis
- French and Arabic support
- Algerian Darija support
- multilingual recommendation generation
- stronger aspect classification
- database integration
- authentication and multi-user support
- production deployment
- automated monitoring of review sources

## License

See [`LICENSE`](LICENSE).

## Author

**Master's Graduation Project — E-Reputation AI**

---

If you use or extend this project, please preserve the academic and technical context of the original work.
