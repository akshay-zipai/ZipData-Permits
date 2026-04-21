# 🏠 House Renovation Advisor API

AI-powered renovation suggestions tailored to your **location** and **space** — with optional DALL-E 3 visualizations.

## Features

- 🤖 GPT-4o powered renovation suggestions
- 🎨 DALL-E 3 visualization generation
- 📍 Location-aware recommendations (climate, culture, materials)
- 💰 Budget tiers: Budget / Mid-range / Premium
- 🐳 Docker ready

---

## Project Structure

```
house-reno-api/
├── app/
│   ├── config/
│   │   └── settings.py       # Env-based config via pydantic-settings
│   ├── prompts/
│   │   └── renovation.py     # System + user prompts for LLM
│   ├── routes/
│   │   ├── health.py         # GET /health
│   │   └── renovation.py     # POST /renovation/suggest
│   ├── services/
│   │   └── llm_service.py    # OpenAI calls (chat + image)
│   ├── models.py             # Pydantic request/response models
│   └── main.py               # FastAPI app entrypoint
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### 1. Setup environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

API is live at: http://localhost:8000

### 3. Run locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## API Usage

### POST `/renovation/suggest`

```bash
curl -X POST http://localhost:8000/renovation/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "place": "Jaipur, India",
    "house_part": "living room",
    "query": "I want a mix of traditional Rajasthani and modern design",
    "generate_image": true
  }'
```

### Sample Response

```json
{
  "place": "Jaipur, India",
  "house_part": "living room",
  "summary": "Jaipur's rich heritage calls for a blend of vibrant Rajasthani craftsmanship...",
  "suggestions": [
    {
      "title": "Pink City Palace Fusion",
      "description": "Blend traditional jharokha windows with clean modern furniture...",
      "style": "Indo-Modern",
      "budget_tier": "Mid-range",
      "key_materials": ["Marble", "Teak wood", "Jaali screens"],
      "estimated_duration": "3-5 weeks",
      "pros": ["Culturally rich", "Great resale value"],
      "local_tip": "Source sandstone and blue pottery locally from Sanganer..."
    }
  ],
  "image_url": "https://...",
  "image_prompt": "..."
}
```

### GET `/health`

```bash
curl http://localhost:8000/health
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Chat model |
| `OPENAI_IMAGE_MODEL` | `dall-e-3` | Image model |
| `IMAGE_SIZE` | `1024x1024` | DALL-E image size |
| `IMAGE_QUALITY` | `standard` | `standard` or `hd` |
| `MAX_SUGGESTIONS` | `5` | Number of suggestions to return |

---

## Interactive Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
