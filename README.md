# Lab Readme

**Source:** `lost-50.csv`  
**Total Items:** 70  
**Total Categories:** 17  
**Date:** April 7, 2026

---

## Project Structure

```
4210-wk-12/

├── app.py                           # Streamlit UI
├── .env                             # OPENAI_API_KEY (not committed)
├── .env.example                     # template for collaborators
├── .gitignore                       # excludes .env, __pycache__, .DS_Store
├── models/
│   ├── tfidf_logreg.joblib          # saved sklearn pipeline
│   └── llm_classifier_config.json  # saved LLM categories + prompt
├── .venv/
│   └── env.sample                   # venv-scoped env template
└── readme.md
```

---


## Category Counts

| Count | Category |
|------:|----------|
| 20 | Electronics |
| 7 | Clothing |
| 6 | Keys, Wallets and Other Personal Accessories |
| 5 | Housewares |
| 5 | Toys and Pets |
| 4 | Luggage, Travel Equipment |
| 4 | Prescription Drugs and Medical Equipment |
| 4 | Toiletries and Hair Products |
| 3 | Disney Parks Products |
| 3 | Cases and Containers |
| 3 | Money and Gift Cards |
| 1 | Eyewear |
| 1 | Footwear |
| 1 | IDs, Drivers Licenses, Credit Cards and Passports |
| 1 | Baby or Child Item |
| 1 | Jewelry |
| 1 | Bottles, Cups and Mugs |

```mermaid
graph TD
    LF[Lost and Found - 70 Items]
    LF --> E[Electronics — 20]
    LF --> C[Clothing — 7]
    LF --> K[Keys Wallets and Accessories — 6]
    LF --> H[Housewares — 5]
    LF --> T[Toys and Pets — 5]
    LF --> L[Luggage Travel Equipment — 4]
    LF --> P[Prescription Drugs and Medical — 4]
    LF --> TO[Toiletries and Hair Products — 4]
    LF --> D[Disney Parks Products — 3]
    LF --> CC[Cases and Containers — 3]
    LF --> M[Money and Gift Cards — 3]
    LF --> OTHER[Other Categories — 6]
```

---

## Classifiers

### Model 1 — TF-IDF + Logistic Regression (`classifier.py`)

Traditional ML pipeline trained on the dataset.

| Setting | Value |
|---------|-------|
| Vectorizer | TF-IDF, unigrams + bigrams, log-scaled TF |
| Model | Logistic Regression, `class_weight=balanced` |
| Cross-validation | 5-fold accuracy: **49% ± 13%** |
| Test split accuracy | **50%** (14 items) |
| Saved model | `models/tfidf_logreg.joblib` |

> Low accuracy is expected — 70 items across 17 categories means some classes have only 1 training example.

```mermaid
graph LR
    CSV[lost-50.csv] --> TFIDF[TF-IDF Vectorizer]
    TFIDF --> LR[Logistic Regression]
    LR --> PRED[Predicted Category]
    LR --> PROB[Class Probabilities]
```


---

### Model 2 — GPT-4o-mini LLM (`classifier_llm.py`)

Zero-shot classification using OpenAI API. All 17 categories are injected into the system prompt so the model can only return a valid label.

| Setting | Value |
|---------|-------|
| Model | `gpt-4o-mini` |
| Temperature | 0 (deterministic) |
| Strategy | Zero-shot with constrained category list |
| API key | loaded from `.env` → `OPENAI_API_KEY` |
| Saved config | `models/llm_classifier_config.json` |

```mermaid
graph LR
    DESC[Item Description] --> SYS[System Prompt + Categories]
    SYS --> GPT[GPT-4o-mini]
    GPT --> CAT[Predicted Category]
```


---

## Streamlit App (`app.py`)

Interactive UI to classify free-text item descriptions using either model.

```mermaid
graph LR
    UI[Streamlit UI] --> SEL[Model Selector]
    SEL --> ML[TF-IDF + LogReg]
    SEL --> LLM[GPT-4o-mini]
    ML --> M1[Confidence Score\nTop-5 Probabilities\nBar Chart]
    LLM --> M2[Latency\nValid Category Check\nMatching Dataset Items]
```

**Launch:**
```bash
streamlit run app.py
```

App runs at: http://localhost:8501

---

## Load Saved Model for Testing

```python
import joblib
pipeline = joblib.load("models/tfidf_logreg.joblib")
pipeline.predict(["black leather wallet"])
# → ['Keys, Wallets and Other Personal Accessories']
```

---

## Setup & Installation

```bash
# Clone the repo

cd 4210-wk12

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas scikit-learn joblib openai python-dotenv streamlit

# Add your OpenAI key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

---


### What is committed vs excluded

| File | Status |
|------|--------|
| `app.py` | ✅ committed |
| `readme.md` | ✅ committed |
| `models/` | ✅ committed |
| `.env.example` | ✅ committed |
| `.env` | ❌ gitignored — never committed |
| `.DS_Store` | ❌ gitignored |
| `__pycache__/` | ❌ gitignored |

```mermaid
graph LR
    DEV[Local Dev] -->|git push| GH[GitHub Repo]
    GH -->|git clone| COL[Collaborator]
    COL --> ENV[cp .env.example .env]
    ENV --> RUN[streamlit run app.py]
```
