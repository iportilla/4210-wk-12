"""
Lost & Found Classifier — Streamlit UI
Supports: TF-IDF + Logistic Regression  |  GPT-4o-mini (LLM)
"""

import os
import json
import time
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lost & Found Classifier",
    page_icon="🔍",
    layout="centered",
)

# ── Load dataset ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("lost-50.csv")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Ticket", "Category"])
    df["Ticket"] = df["Ticket"].str.strip()
    return df

df = load_data()
CATEGORIES = sorted(df["Category"].unique().tolist())

# ── Load sklearn model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_sklearn_model():
    return joblib.load("models/tfidf_logreg.joblib")

# ── Load LLM config ────────────────────────────────────────────────────────────
@st.cache_data
def load_llm_config():
    return json.loads(Path("models/llm_classifier_config.json").read_text())

# ── LLM classify ──────────────────────────────────────────────────────────────
def llm_classify(description: str, config: dict) -> tuple[str, float]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": description},
        ],
        temperature=0,
        max_tokens=50,
    )
    latency = time.perf_counter() - t0
    category = response.choices[0].message.content.strip()
    return category, latency

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 Lost & Found Classifier")
st.caption(f"Dataset: **{len(df)} items** across **{len(CATEGORIES)} categories**")

st.divider()

# ── Model selector ─────────────────────────────────────────────────────────────
model_choice = st.radio(
    "Select classifier",
    options=["TF-IDF + Logistic Regression", "GPT-4o-mini (LLM)"],
    horizontal=True,
)

with st.expander("ℹ️ About this model"):
    if model_choice == "TF-IDF + Logistic Regression":
        st.markdown("""
- **Type:** Traditional ML (scikit-learn)
- **Vectorizer:** TF-IDF with unigrams + bigrams
- **Model:** Logistic Regression (`class_weight=balanced`)
- **CV Accuracy:** ~49% (5-fold) — limited by small dataset
- **Speed:** Instant (local, no API call)
        """)
    else:
        st.markdown("""
- **Type:** Large Language Model (OpenAI API)
- **Model:** `gpt-4o-mini`
- **Strategy:** Zero-shot classification with constrained category list
- **Speed:** ~0.5–1.5s per call (API latency)
- **Cost:** ~$0.00003 per prediction
        """)

st.divider()

# ── Input ──────────────────────────────────────────────────────────────────────
description = st.text_input(
    "Enter item description",
    placeholder="e.g. Black leather wallet, AirPods case, Dyson Airwrap...",
)

predict_btn = st.button("Classify", type="primary", disabled=not description.strip())

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn and description.strip():
    st.divider()

    if model_choice == "TF-IDF + Logistic Regression":
        pipeline = load_sklearn_model()

        t0 = time.perf_counter()
        prediction = pipeline.predict([description])[0]
        probas = pipeline.predict_proba([description])[0]
        latency = time.perf_counter() - t0

        confidence = probas.max()
        classes = pipeline.classes_

        # Result
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Category", prediction)
        col2.metric("Confidence", f"{confidence:.0%}")
        col3.metric("Latency", f"{latency*1000:.1f} ms")

        st.divider()

        # Top-5 probabilities
        st.subheader("Top 5 category probabilities")
        proba_df = (
            pd.DataFrame({"Category": classes, "Probability": probas})
            .sort_values("Probability", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
        proba_df["Probability %"] = (proba_df["Probability"] * 100).round(1)
        st.dataframe(
            proba_df[["Category", "Probability %"]],
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(
            proba_df.set_index("Category")["Probability"],
            use_container_width=True,
        )

        # Dataset distribution
        st.divider()
        st.subheader("Training data distribution")
        dist = df["Category"].value_counts().reset_index()
        dist.columns = ["Category", "Count"]
        st.dataframe(dist, use_container_width=True, hide_index=True)

    else:
        # LLM path
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or not api_key.startswith("sk-"):
            st.error("OPENAI_API_KEY not found or invalid in your .env file.")
            st.stop()

        config = load_llm_config()

        with st.spinner("Calling GPT-4o-mini..."):
            try:
                prediction, latency = llm_classify(description, config)
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        valid = prediction in CATEGORIES

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Category", prediction)
        col2.metric("Valid Category", "✅ Yes" if valid else "⚠️ No")
        col3.metric("Latency", f"{latency:.2f}s")

        if not valid:
            st.warning(f"The model returned '{prediction}' which is not in the known category list.")

        st.divider()

        # Show which category it matched in the dataset
        if valid:
            st.subheader(f"Items in dataset under '{prediction}'")
            matches = df[df["Category"] == prediction][["Ticket"]].reset_index(drop=True)
            matches.index += 1
            st.dataframe(matches, use_container_width=True)

        st.divider()
        st.subheader("All known categories")
        dist = df["Category"].value_counts().reset_index()
        dist.columns = ["Category", "Count"]
        st.dataframe(dist, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("ORGN 4210 — Week 12 Lab · Lost & Found Classifier")
