"""Step 2 of fine-tuning — upload the training data to Google Cloud Storage.

Vertex AI reads tuning data from a GCS bucket, so we create one (if it
doesn't exist) and upload train.jsonl + validation.jsonl into it.

Run:
    python finetuning/upload_to_bucket.py

Prints the gs:// URIs to paste into the Vertex tuning console (Step 2,
"Tuning dataset").
"""

import os
import sys

from google.cloud import storage

# Make the project's config.py importable when run from finetuning/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

BUCKET_NAME = f"{config.GCP_PROJECT}-finetuning-data"
BUCKET_LOCATION = "us-central1"   # match the tuning region
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PREFIX = "headline-tuning"        # a "folder" inside the bucket
FILES = ["train.jsonl", "validation.jsonl"]


def main() -> None:
    client = storage.Client(project=config.GCP_PROJECT)

    # 1. Create the bucket if it doesn't already exist.
    bucket = client.bucket(BUCKET_NAME)
    if bucket.exists():
        print(f"Bucket gs://{BUCKET_NAME} already exists — reusing it.")
    else:
        bucket = client.create_bucket(bucket, location=BUCKET_LOCATION)
        print(f"Created bucket gs://{BUCKET_NAME} in {BUCKET_LOCATION}.")

    # 2. Upload each file into the bucket.
    for fname in FILES:
        local_path = os.path.join(DATA_DIR, fname)
        blob = bucket.blob(f"{PREFIX}/{fname}")
        blob.upload_from_filename(local_path)
        print(f"Uploaded {fname} -> gs://{BUCKET_NAME}/{PREFIX}/{fname}")

    # 3. Print the URIs for the tuning console.
    print("\nPaste these into the Vertex tuning console (Step 2 — Tuning dataset):")
    print(f"  Training dataset   : gs://{BUCKET_NAME}/{PREFIX}/train.jsonl")
    print(f"  Validation dataset : gs://{BUCKET_NAME}/{PREFIX}/validation.jsonl")


if __name__ == "__main__":
    main()
