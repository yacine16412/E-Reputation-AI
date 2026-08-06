"""Streamlit entry point for E-Reputation AI."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics import build_business_aspect_table, build_pareto_data, summarize_priority_counts
from aspects import build_aspect_analysis
from config import (
    APP_DESCRIPTION, APP_TITLE,
    DEFAULT_RECOMMENDATION_MAX_INPUT_LENGTH, DEFAULT_RECOMMENDATION_MAX_NEW_TOKENS,
    DEFAULT_RECOMMENDATION_NUM_BEAMS, DEFAULT_SENTIMENT_BATCH_SIZE,
    DEFAULT_SENTIMENT_MAX_LENGTH, DEFAULT_TOP_K_REVIEWS,
    RECOMMENDATION_MODEL_DIR, SENTIMENT_MODEL_DIR,
)
from export import build_analysis_exports
from preprocessing import clean_comment
from recommendations import generate_recommendation, load_recommendation_model, select_representative_reviews
from sentiment import load_sentiment_model, predict_sentiment_batch

st.set_page_config(page_title=APP_TITLE, page_icon="🏨", layout="wide")
st.title(APP_TITLE)
st.caption(APP_DESCRIPTION)

@st.cache_resource(show_spinner=False)
def get_sentiment_resources():
    return load_sentiment_model(SENTIMENT_MODEL_DIR)

@st.cache_resource(show_spinner=False)
def get_recommendation_resources():
    return load_recommendation_model(RECOMMENDATION_MODEL_DIR)

def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")

def run_sentiment_analysis(dataframe, review_column, max_reviews, batch_size):
    tokenizer, model, device = get_sentiment_resources()
    result = dataframe.head(max_reviews).copy()
    result["clean_text"] = result[review_column].fillna("").astype(str).map(clean_comment)
    labels, confidences = predict_sentiment_batch(
        result["clean_text"].tolist(), tokenizer, model, device,
        batch_size=batch_size, max_length=DEFAULT_SENTIMENT_MAX_LENGTH,
    )
    result["sentiment"] = labels
    result["confidence"] = confidences
    return result

def generate_aspect_recommendations(negative_reviews, aspect_summary):
    columns = ["aspect", "representative_reviews", "issue_summary", "recommendation"]
    if aspect_summary is None or aspect_summary.empty:
        return pd.DataFrame(columns=columns)

    tokenizer, model, device = get_recommendation_resources()
    rows = []
    for aspect in aspect_summary["aspect"].tolist():
        evidence = select_representative_reviews(
            negative_reviews, aspect, top_k=DEFAULT_TOP_K_REVIEWS
        )
        issue, recommendation = generate_recommendation(
            aspect, evidence, tokenizer, model, device,
            max_input_length=DEFAULT_RECOMMENDATION_MAX_INPUT_LENGTH,
            max_new_tokens=DEFAULT_RECOMMENDATION_MAX_NEW_TOKENS,
            num_beams=DEFAULT_RECOMMENDATION_NUM_BEAMS,
        )
        rows.append({
            "aspect": aspect,
            "representative_reviews": " | ".join(evidence),
            "issue_summary": issue,
            "recommendation": recommendation,
        })
    return pd.DataFrame(rows, columns=columns)

uploaded_file = st.file_uploader("Upload hotel reviews CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to start the analysis.")
    st.stop()

data = read_uploaded_csv(uploaded_file)

if data.empty:
    st.warning("The uploaded CSV is empty.")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

columns = list(data.columns)
default_index = columns.index("comment") if "comment" in columns else 0

review_column = st.selectbox(
    "Select the column containing customer reviews",
    options=columns,
    index=default_index,
)

max_reviews = st.number_input(
    "Maximum number of reviews to analyze",
    min_value=1, max_value=len(data), value=len(data), step=1,
)

batch_size = st.number_input(
    "Batch size",
    min_value=1, max_value=128, value=DEFAULT_SENTIMENT_BATCH_SIZE, step=1,
)

if st.button("Run analysis", type="primary"):
    with st.spinner("Running sentiment analysis..."):
        analyzed = run_sentiment_analysis(
            data, review_column, int(max_reviews), int(batch_size)
        )

    negative_reviews = analyzed[
        analyzed["sentiment"].str.lower() == "negative"
    ].copy()
    positive_count = int(
        (analyzed["sentiment"].str.lower() == "positive").sum()
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews analyzed", len(analyzed))
    c2.metric("Positive", positive_count)
    c3.metric("Negative", len(negative_reviews))

    st.subheader("Sentiment results")
    st.dataframe(
        analyzed[[review_column, "sentiment", "confidence", "clean_text"]],
        use_container_width=True,
    )

    if negative_reviews.empty:
        st.warning("No negative reviews were detected.")
        st.stop()

    with st.spinner("Detecting negative aspects..."):
        negative_with_aspects, aspect_summary = build_aspect_analysis(
            negative_reviews
        )

    st.subheader("Negative aspects")
    st.dataframe(aspect_summary, use_container_width=True)

    business_aspects = build_business_aspect_table(
        aspect_summary, negative_review_count=len(negative_reviews)
    )

    st.subheader("Business priorities")
    st.dataframe(business_aspects, use_container_width=True)

    priority_counts = summarize_priority_counts(business_aspects)
    st.bar_chart(priority_counts.set_index("priority")["count"])

    pareto = build_pareto_data(business_aspects)
    if not pareto.empty:
        st.subheader("Pareto analysis")
        st.bar_chart(pareto.set_index("aspect")["mentions"])

    with st.spinner("Generating recommendations with FLAN-T5..."):
        recommendations = generate_aspect_recommendations(
            negative_with_aspects, aspect_summary
        )

    st.subheader("Recommendations")
    st.dataframe(recommendations, use_container_width=True)

    exports = build_analysis_exports(
        analyzed_reviews=analyzed,
        aspect_summary=aspect_summary,
        recommendations=recommendations,
    )

    st.subheader("Export")
    for filename, payload in exports.items():
        st.download_button(
            label=f"Download {filename}",
            data=payload,
            file_name=filename,
            mime="text/csv",
            key=f"download-{filename}",
        )
