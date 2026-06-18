# Fine-tuning Gemini for News Headline Generation

Supervised fine-tuning of `gemini-2.5-flash` on Vertex AI to turn a news
description into a concise headline. This folder holds everything needed to
build the training data and start the tuning job.

## The task

| | |
|---|---|
| Input (user)  | a news short description |
| Output (model)| the article's headline |
| Base model    | `gemini-2.5-flash` (tunable on Vertex AI) |
| Dataset       | HuffPost News Category Dataset (`heegyu/news-category-dataset`) |
| Examples      | 5,000 (4,500 train / 500 validation) |
| Method        | Supervised fine-tuning (LoRA — parameter-efficient) |
| Region        | `us-central1` |

## How fine-tuning works (short version)

The base model is frozen; Vertex trains a small **LoRA adapter** on our
examples. For each example it runs forward pass → compute loss against the
target headline → backpropagation → gradient descent, over a few **epochs**,
using the validation split to watch for **overfitting**. The result is a
tuned model that writes headlines in the trained style.

## Steps

### 1. Install the prep dependencies
```bash
pip install -r finetuning/requirements.txt
```

### 2. Build the training data
```bash
python finetuning/prepare_dataset.py
```
Downloads the dataset, filters and converts it to Vertex's JSONL format, and
writes `finetuning/data/train.jsonl` and `validation.jsonl`. Each line is a
mini-conversation:
```json
{"systemInstruction":{"role":"system","parts":[{"text":"You are a news headline writer..."}]},
 "contents":[{"role":"user","parts":[{"text":"<description>"}]},
             {"role":"model","parts":[{"text":"<headline>"}]}]}
```

### 3. Upload to Cloud Storage
```bash
python finetuning/upload_to_bucket.py
```
Creates a bucket and uploads the two files. Prints the `gs://` URIs.

### 4. Start the tuning job (Vertex AI console)
Vertex AI → Tuning → Create tuned model:
- Supervised fine-tuning → Tune a foundation model
- Base model `gemini-2.5-flash`, region `us-central1`
- Training dataset = the `gs://…/train.jsonl` URI; validation = the `…/validation.jsonl` URI
- Epochs = 3 (defaults for the rest) → **Start tuning**

Runs 30 min – a few hours. Watch the loss curves on the job page.

### 5. Wire the tuned model into the app
When the job finishes, copy the tuned model's resource name
(`projects/<num>/locations/us-central1/models/<id>`) into `.env`:
```
TUNED_GEMINI_MODEL=projects/<num>/locations/us-central1/models/<id>
```
The app's **Fine-tuned** page then shows base vs fine-tuned headlines side
by side.

## Cost

Billed per training token ≈ examples × tokens/example × epochs. ~5,000
examples × 3 epochs is on the order of a few euros (well within the
education credit). To test cheaply first, lower `NUM_EXAMPLES` in
`prepare_dataset.py` to ~1,000.
