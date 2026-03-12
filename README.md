# California Permit RAG System

A production-ready FastAPI backend for answering permit-related questions using web crawling, RAG (Retrieval-Augmented Generation), and LLM inference.

## Architecture

```
ca_permit_rag/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── crawl.py          # Web crawling endpoints
│   │       ├── rag.py            # RAG query endpoints
│   │       ├── llm.py            # LLM inference endpoints
│   │       └── websocket.py      # WebSocket for permit Q&A
│   ├── core/
│   │   ├── config.py             # App configuration
│   │   └── logging.py            # Logging setup
│   ├── models/
│   │   ├── requests.py           # Pydantic request models
│   │   └── responses.py          # Pydantic response models
│   ├── services/
│   │   ├── crawling/
│   │   │   └── crawler.py        # Web scraping service
│   │   ├── embedding/
│   │   │   └── embedder.py       # SBERT embedding service
│   │   ├── llm/
│   │   │   └── generator.py      # Gemma LLM service
│   │   └── rag/
│   │       ├── retriever.py      # Hybrid BM25 + vector retriever
│   │       └── pipeline.py       # RAG pipeline orchestrator
│   └── utils/
│       ├── permit_portals.py     # Portal lookup utility
│       └── text_processing.py    # Text chunking utils
├── prompts/
│   ├── qa_system.txt             # System prompt for Q&A
│   └── rag_context.txt           # RAG context prompt template
├── data/
│   └── permit_portals.json       # CA county permit portal URLs
├── tests/
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Services

- **Crawling Service**: Scrapes permit portal websites by ZIP code / county
- **Embedding Service**: SBERT `all-MiniLM-L12-v2` for semantic embeddings (swappable)
- **LLM Service**: Gemma 3 4B (via Ollama or HuggingFace) for answer generation (swappable)
- **RAG Service**: Hybrid BM25 + ChromaDB vector search for best retrieval

## Running

```bash
# Copy env file
cp .env.example .env

# Start with Docker Compose
docker compose up --build

# API available at http://localhost:8000
# WebSocket at ws://localhost:8000/ws/permit-qa
```

## API Endpoints

- `POST /api/v1/crawl/scrape` — Scrape a permit portal by ZIP or county
- `POST /api/v1/rag/index` — Index scraped content into vector DB
- `POST /api/v1/rag/query` — Query with hybrid retrieval
- `POST /api/v1/llm/generate` — Raw LLM generation
- `WS  /ws/permit-qa` — WebSocket for interactive permit Q&A
