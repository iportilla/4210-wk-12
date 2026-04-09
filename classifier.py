"""
Lost & Found Item Classifier
Uses TF-IDF + Logistic Regression to predict category from ticket description.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("lost-50.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Ticket", "Category"])
df["Ticket"] = df["Ticket"].str.strip()

X = df["Ticket"]
y = df["Category"]

print(f"Dataset: {len(df)} items, {y.nunique()} categories\n")

# ── Build pipeline ─────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=1,
        sublinear_tf=True,    # log-scale TF to dampen frequency
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # handle imbalanced categories
        random_state=42,
    )),
])

# ── Cross-validation (leave-one-out friendly with small dataset) ───────────────
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"5-fold CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
print(f"Fold scores: {[round(s, 2) for s in cv_scores]}\n")

# ── Train/test split for classification report ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Classification Report (test split):")
print(classification_report(y_test, y_pred, zero_division=0))

# ── Train on full dataset for inference ───────────────────────────────────────
pipeline.fit(X, y)

# ── Save model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/tfidf_logreg.joblib")
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved → {MODEL_PATH}\n")

# ── Interactive prediction ─────────────────────────────────────────────────────
print("\n── Predict a category ──────────────────────────────────────────────────")
print("Type a lost item description (or 'quit' to exit):\n")

while True:
    try:
        text = input("Item: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not text or text.lower() == "quit":
        break
    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text]).max()
    print(f"  → Category: {pred}  (confidence: {proba:.0%})\n")
