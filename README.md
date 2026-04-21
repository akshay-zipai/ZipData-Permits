# 🏡 California Permit & Renovation Agent

A conversational AI agent that combines:

- **🏛️ CA Permit RAG** — answers California building permit questions from an offline dataset of 20,000+ records scraped from county permit portals across all 58 CA counties
- **🎨 Renovation Advisor** — AI-powered renovation suggestions with DALL-E design visualisations

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                     │
│    Chat · Quick-Reply Chips · Inline Reno Cards         │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (REST)
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend Agent                   │
│                                                          │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │   Permit RAG        │  │   Renovation AI          │  │
│  │  Offline JSONL KB   │  │  Text: OpenAI / Bedrock  │  │
│  │  20 k+ CA records   │  │  Image: DALL-E / Titan   │  │
│  │  Lexical retrieval  │  │  5 style/budget options  │  │
│  └─────────────────────┘  └──────────────────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │          agent/llm.py  (LLM abstraction)        │    │
│  │   ENVIRONMENT=local  → OpenAI GPT               │    │
│  │   ENVIRONMENT=production → AWS Bedrock Converse  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Conversation flow

```
COLLECT_LOCATION  (ZIP / county)
       │
       ▼
COLLECT_PERMIT_QUESTION ◄────────────────────────────┐
       │                                             │
       ▼                                             │
ANSWERING_PERMIT  (RAG over offline JSONL KB)        │
       │                                             │
       ▼                                             │
PERMIT_FOLLOWUP ──► new question ────────────────────┘
       │
       ▼  (user asks for renovation)
COLLECT_RENO_AREA
       │
       ▼
COLLECT_RENO_PREFS
       │
       ▼
GENERATING_RENO  (LLM suggestions + image)
       │
       ▼
RENO_FOLLOWUP ──► permits ──► COLLECT_PERMIT_QUESTION
               └──► more reno ──► COLLECT_RENO_AREA
```

---

## Quick start

### Option 1 — Docker Compose (recommended)

```bash
# 1. Copy and edit the env file
cp .env.example .env

# For LOCAL mode: set ENVIRONMENT=local and add OPENAI_API_KEY
# For PROD mode:  set ENVIRONMENT=production and add AWS credentials

# 2. Build and run
docker compose up --build

# Frontend: http://localhost:8501
# API docs: http://localhost:8000/docs
# Health:   http://localhost:8000/health
```

### Option 2 — Local Python (no Docker)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install streamlit==1.41.1     # frontend dependency

cp .env.example .env
# edit .env — at minimum set OPENAI_API_KEY for local mode

# Terminal 1 — API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend
API_BASE_URL=http://localhost:8000 streamlit run app.py
```

---

## Environment variables

### Shared (both modes)

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `local` | `local` = OpenAI · `production` = Bedrock |
| `RAG_TOP_K` | `5` | Permit KB chunks per answer |
| `MAX_SUGGESTIONS` | `5` | Renovation suggestions |
| `GENERATE_IMAGES` | `true` | `false` skips image generation |
| `IMAGE_SIZE` | `1024x1024` | DALL-E 3 output size |
| `IMAGE_QUALITY` | `standard` | `standard` or `hd` |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per generation |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |

### Local mode (`ENVIRONMENT=local`)

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `OPENAI_MODEL` | No | Default: `gpt-4o-mini` |
| `OPENAI_IMAGE_MODEL` | No | Default: `dall-e-3` |

### Production mode (`ENVIRONMENT=production`)

| Variable | Required | Description |
|---|---|---|
| `BEDROCK_REGION` | ✅ | AWS region, e.g. `us-east-1` |
| `BEDROCK_MODEL_ID` | ✅ | Converse-API model ID (see options below) |
| `AWS_ACCESS_KEY_ID` | If no IAM role | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | If no IAM role | AWS secret key |
| `AWS_SESSION_TOKEN` | If using STS | Session token |
| `BEDROCK_IMAGE` | No | `true` → Titan Image Generator; `false` (default) → DALL-E |
| `OPENAI_API_KEY` | If `BEDROCK_IMAGE=false` | Still used for DALL-E images in prod |

#### Supported Bedrock model IDs

All models below support the Bedrock **Converse API** (unified interface):

```
anthropic.claude-3-haiku-20240307-v1:0        # fast, cheap
anthropic.claude-3-sonnet-20240229-v1:0       # balanced
anthropic.claude-3-5-sonnet-20241022-v2:0     # best quality
amazon.nova-lite-v1:0                         # Amazon native, very fast
amazon.nova-pro-v1:0                          # Amazon native, high quality
meta.llama3-8b-instruct-v1:0                  # open-source
mistral.mistral-7b-instruct-v0:2              # open-source
```

Enable the model in **AWS Console → Bedrock → Model access** before use.

---

## Data files

All permit data is served **100% offline** — no internet needed for retrieval:

| File | Description |
|---|---|
| `data/bedrock_kb_by_zip.jsonl` | 20,053 permit records keyed by ZIP code |
| `data/bedrock_kb.jsonl` | Permit records keyed by county |
| `data/california_permit_mapping.json` | County → permit type mapping |
| `data/permit_portals.json` | County permit portal URLs |
| `data/Sample_que.json` | Sample permit questions for testing |

Only the **LLM generation calls** and **image generation** require internet/API access.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + backend info |
| `POST` | `/session/start` | Start a new conversation |
| `POST` | `/chat` | Send a message, get a response |
| `POST` | `/session/reset` | Reset session state |
| `GET` | `/counties` | List all CA counties in KB |
| `GET` | `/counties/{county}/zips` | List ZIP codes for a county |

Interactive docs: `http://localhost:8000/docs`

---

## Project structure

```
reno-permit-agent/
├── agent/
│   ├── __init__.py
│   ├── config.py        ← Settings (pydantic-settings, reads .env)
│   ├── llm.py           ← LLM abstraction: OpenAI ↔ Bedrock
│   ├── agent.py         ← State machine + conversation logic
│   └── permit_kb.py     ← Offline JSONL KB loader + retrieval
├── frontend/
│   ├── app.py           ← Streamlit UI (theme-adaptive)
│   └── requirements.txt
├── data/                ← Offline CA permit dataset (shipped in zip)
├── prompts/             ← Optional prompt override .txt files
├── main.py              ← FastAPI entry point
├── requirements.txt     ← API dependencies (incl. boto3 + openai)
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.example         ← Copy to .env and configure
└── README.md
```

---

## Cost estimates (approximate)

| Operation | Model | Cost |
|---|---|---|
| Permit Q&A | GPT-4o-mini | ~$0.001–0.003 |
| Permit Q&A | Claude 3 Haiku (Bedrock) | ~$0.001–0.002 |
| Renovation suggestions | GPT-4o-mini | ~$0.005–0.01 |
| Renovation suggestions | Claude 3 Haiku (Bedrock) | ~$0.004–0.008 |
| DALL-E 3 image (standard) | OpenAI | ~$0.04 |
| Titan Image v2 (Bedrock) | AWS | ~$0.01 |

To reduce costs: set `GENERATE_IMAGES=false`, use `gpt-3.5-turbo` / `nova-lite`, or set `IMAGE_QUALITY=standard`.

---

## Troubleshooting

**`OPENAI_API_KEY is not set`** — add it to your `.env` file.

**Bedrock `AccessDeniedException`** — enable the model in AWS Console → Bedrock → Model access.

**Bedrock `ValidationException`** — check `BEDROCK_MODEL_ID` matches exactly the model string shown in AWS Console (including version suffix).

**Slow responses** — DALL-E image generation takes 10–20 s. Set `GENERATE_IMAGES=false` for faster demos.

**`No permit data found`** — verify `data/bedrock_kb_by_zip.jsonl` exists: `curl http://localhost:8000/health` should show `kb_records > 0`.
