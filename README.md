# DoubleChat — Multi-LLM Comparison Lab

**SRH Advanced Computer Science — Final Exam Project (Topic K)**

A Streamlit web application that compares large language models —
**Google Gemini** (served through **Vertex AI** on Google Cloud) and
**Anthropic Claude** (via the Anthropic API) — side by side on the same
prompt, measuring answer quality, latency, and token usage.
OpenAI ChatGPT is covered analytically in the written comparison report.

## Architecture

```
┌─────────────────────┐
│  Browser            │
│  (localhost:8501)   │
└─────────┬───────────┘
          │ HTTP
┌─────────▼───────────┐      ┌──────────────────────────────┐
│  app.py             │      │ chatbots/gemini_bot.py       │──▶ Vertex AI
│  (nav + sidebar)    │─────▶│ auth: ADC (gcloud login)     │    (GCP, billed to
│                     │      └──────────────────────────────┘     project credit)
│  views/             │      ┌──────────────────────────────┐
│   compare / chat /  │─────▶│ chatbots/claude_bot.py       │──▶ Anthropic API
│   history           │      │ auth: API key from .env      │    (prepaid credit)
│                     │      └──────────────────────────────┘
│  config.py          │      ┌──────────────────────────────┐
│  (.env settings)    │─────▶│ storage/firestore_db.py      │──▶ Cloud Firestore
└─────────────────────┘      │ auth: ADC · CRUD + timings   │    (GCP, europe-west3)
                             └──────────────────────────────┘
        LLM wrappers return the same dataclass: chatbots/base.py::LLMResponse
        Firestore functions return (result, latency_ms) for the UI to display
```

## Project structure

```
final-exam-project/
├── app.py                 # Entry point — page config + sidebar navigation
├── config.py              # All settings; loads .env exactly once
├── check_setup.py         # Verifies LLMs + Firestore work (run after setup)
├── chatbots/              # LLM provider wrappers
│   ├── __init__.py        # Exports: ask_*, chat_*, generate_image, transcribe_audio
│   ├── base.py            # LLMResponse dataclass (shared response shape)
│   ├── gemini_bot.py      # Gemini via Vertex AI: text, image, audio (ADC auth)
│   └── claude_bot.py      # Claude via Anthropic API (key auth)
├── storage/               # Persistence layer
│   ├── __init__.py        # Exports the Firestore CRUD functions
│   └── firestore_db.py    # Conversation save/load/list/delete + timings
├── benchmark/             # Benchmark engine
│   ├── __init__.py
│   └── suite.py           # Preset prompts, cost estimation, run_benchmark()
├── views/                 # One render() function per UI page
│   ├── __init__.py
│   ├── playground.py      # Tabs: Compare (text) · Image · Voice
│   ├── chat.py            # Multi-turn chat, auto-saved to Firestore
│   ├── benchmark.py       # Auto-generated comparison table (latency/tokens/cost)
│   └── history.py         # Browse / reload / delete saved conversations
├── requirements.txt       # Python dependencies
├── .env.example           # Template for required secrets/settings
├── .env                   # Real secrets — gitignored, never committed
└── .gitignore
```

## Setup on a new machine

### Prerequisites
- Python 3.10+ (`python --version`)
- A Google account with access to a GCP project that has billing enabled
  and the **Vertex AI API** enabled
- An Anthropic API key with credit (https://console.anthropic.com)
- Google Cloud CLI (https://cloud.google.com/sdk/docs/install)

### 1. Get the code and create a virtual environment

```powershell
cd path\to\final-exam-project
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure secrets

```powershell
copy .env.example .env             # then edit .env with real values
```

- `ANTHROPIC_API_KEY` — from console.anthropic.com → API Keys
- `GCP_PROJECT` — your project ID (`gcloud projects list`)

### 3. Authenticate Google Cloud (one time per machine)

```powershell
gcloud auth login
gcloud auth application-default login
gcloud auth application-default set-quota-project <your-project-id>
gcloud services enable aiplatform.googleapis.com --project <your-project-id>
```

### 4. Verify, then run

```powershell
python check_setup.py              # both providers must print OK
streamlit run app.py               # opens http://localhost:8501
```

## Why two different authentication methods?

| Provider | Method | Where the secret lives |
|---|---|---|
| Claude | Static API key | `.env` file (gitignored) |
| Gemini / Vertex AI | Application Default Credentials (OAuth) | `gcloud` credential store in the user profile |

This is intentional and worth comparing: API keys are simple but
long-lived (must be rotated manually); ADC issues short-lived OAuth
tokens tied to a Google identity — the more secure enterprise pattern.

## Cost model

| Service | Cost |
|---|---|
| Gemini via Vertex AI | ~€0.0002 per query, billed to the GCP education credit |
| Claude (Opus 4.8) | ~$0.005 per query, from prepaid Anthropic credit |
| Firestore (planned) | Free tier (50K reads / 20K writes per day) |

## Roadmap

- [x] Phase 1 — Streamlit skeleton, environment, secrets handling
- [x] Phase 2 — Side-by-side comparison with latency + token metrics
- [x] Phase 3 — Multi-turn chat mode with history persisted to Firestore
- [x] Capabilities — Gemini image generation + speech-to-text
- [x] Phase 4 — Benchmark suite + auto-generated comparison table
- [ ] Phase 5 — Fine-tune Gemini on Vertex AI (customization dimension)
- [ ] Phase 6 — Deploy to Cloud Run (public demo URL)
