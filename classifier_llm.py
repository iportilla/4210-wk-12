"""
Lost & Found Item Classifier — LLM version (OpenAI)
Uses GPT to predict category from ticket description.
API key loaded from .env file.
"""

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Load data and extract known categories ─────────────────────────────────────
df = pd.read_csv("lost-50.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Ticket", "Category"])
df["Ticket"] = df["Ticket"].str.strip()

CATEGORIES = sorted(df["Category"].unique().tolist())

SYSTEM_PROMPT = f"""You are a lost-and-found item classifier.
Given a description of a lost item, respond with ONLY the single most appropriate category from this list:

{chr(10).join(f"- {c}" for c in CATEGORIES)}

Rules:
- Reply with the exact category name, nothing else.
- If unsure, pick the closest match from the list.
- Never invent a new category."""


# ── Save config (categories + prompt) ────────────────────────────────────────
MODEL_PATH = Path("models/llm_classifier_config.json")
MODEL_PATH.parent.mkdir(exist_ok=True)
config = {
    "model": "gpt-4o-mini",
    "categories": CATEGORIES,
    "system_prompt": SYSTEM_PROMPT,
}
MODEL_PATH.write_text(json.dumps(config, indent=2))
print(f"LLM config saved → {MODEL_PATH}\n")


def classify(description: str) -> tuple[str, str]:
    """Return (category, raw_response) for a given item description."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        temperature=0,
        max_tokens=50,
    )
    category = response.choices[0].message.content.strip()
    return category, response.choices[0].finish_reason


# ── Evaluate on the full dataset ───────────────────────────────────────────────
def evaluate():
    print(f"Evaluating {len(df)} items against {len(CATEGORIES)} categories...\n")
    correct = 0
    errors = []

    for _, row in df.iterrows():
        ticket = row["Ticket"]
        true_label = row["Category"]
        predicted, _ = classify(ticket)

        if predicted == true_label:
            correct += 1
        else:
            errors.append((ticket, true_label, predicted))

    accuracy = correct / len(df)
    print(f"Accuracy: {correct}/{len(df)} = {accuracy:.0%}\n")

    if errors:
        print(f"Misclassified ({len(errors)}):")
        for ticket, true, pred in errors:
            print(f"  [{true}] → predicted [{pred}]")
            print(f"    item: {ticket}")
        print()

    return accuracy


# ── Interactive prediction ─────────────────────────────────────────────────────
def interactive():
    print("── Predict a category ──────────────────────────────────────────────────")
    print("Type a lost item description (or 'quit' to exit):\n")
    while True:
        try:
            text = input("Item: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() == "quit":
            break
        category, _ = classify(text)
        print(f"  → Category: {category}\n")


if __name__ == "__main__":
    import sys

    if "--evaluate" in sys.argv:
        evaluate()
    else:
        print(f"Known categories: {len(CATEGORIES)}\n")
        interactive()
