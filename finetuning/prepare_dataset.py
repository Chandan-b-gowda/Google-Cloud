"""Step 1 of fine-tuning — build the training data.

Downloads the HuffPost News Category Dataset and converts it into the JSONL
format Vertex AI requires for supervised fine-tuning of Gemini.

Task: news headline generation.
    input  (user role)  = a news short_description
    output (model role) = the article's real headline   (this is the target)

Run:
    python finetuning/prepare_dataset.py

Produces:
    finetuning/data/train.jsonl        (90% — what the model learns from)
    finetuning/data/validation.jsonl   (10% — held out, to detect overfitting)
"""

import json
import os
import random

from datasets import load_dataset

# --- Settings (tweak these) -------------------------------------------------
DATASET = "heegyu/news-category-dataset"   # HuffPost, ~210k rows, no auth needed
NUM_EXAMPLES = 5000          # total to use. Lower to ~1000 for a cheap first run.
VALIDATION_FRACTION = 0.1    # 10% held out for overfitting checks
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# The fixed instruction shown on every example — it defines the model's job.
SYSTEM_INSTRUCTION = (
    "You are a news headline writer. Given a short description of a news "
    "story, write a single concise, catchy headline in news style. "
    "Return only the headline, with no extra text."
)

# Quality filters — drop examples that are too short/long to be useful.
MIN_DESC_LEN, MAX_DESC_LEN = 30, 1000
MIN_HEADLINE_LEN, MAX_HEADLINE_LEN = 15, 200


def is_good(description: str, headline: str) -> bool:
    """Keep only clean, reasonably-sized pairs."""
    if not description or not headline:
        return False
    if not (MIN_DESC_LEN <= len(description) <= MAX_DESC_LEN):
        return False
    if not (MIN_HEADLINE_LEN <= len(headline) <= MAX_HEADLINE_LEN):
        return False
    return True


def to_gemini_example(description: str, headline: str) -> dict:
    """Convert one (description, headline) pair into Vertex's tuning format.

    Each example is a mini-conversation: the user provides the description,
    and the model's expected reply is the headline.
    """
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": SYSTEM_INSTRUCTION}],
        },
        "contents": [
            {"role": "user", "parts": [{"text": description}]},
            {"role": "model", "parts": [{"text": headline}]},
        ],
    }


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    print(f"Loading {DATASET} (streaming, no full download)...")
    stream = load_dataset(DATASET, split="train", streaming=True)

    # 1. Collect clean examples until we have enough.
    examples: list[dict] = []
    scanned = 0
    for row in stream:
        scanned += 1
        desc = (row.get("short_description") or "").strip()
        head = (row.get("headline") or "").strip()
        if is_good(desc, head):
            examples.append(to_gemini_example(desc, head))
        if len(examples) >= NUM_EXAMPLES:
            break
        if scanned > NUM_EXAMPLES * 10:  # safety stop if data is sparse
            break
    print(f"Scanned {scanned} rows, kept {len(examples)} good examples.")

    # 2. Shuffle, then split into train / validation.
    random.seed(42)  # fixed seed = reproducible split
    random.shuffle(examples)
    n_val = int(len(examples) * VALIDATION_FRACTION)
    validation, train = examples[:n_val], examples[n_val:]

    # 3. Write both JSONL files.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_path = os.path.join(OUTPUT_DIR, "validation.jsonl")
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, validation)

    print(f"Wrote {len(train):>5} examples -> {train_path}")
    print(f"Wrote {len(validation):>5} examples -> {val_path}")
    print("\nSample training line:")
    print(json.dumps(train[0], ensure_ascii=False)[:320], "...")


if __name__ == "__main__":
    main()
