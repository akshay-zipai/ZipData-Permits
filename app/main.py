import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import renovation_router, health_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="🏠 House Renovation Advisor API",
    description=(
        "AI-powered renovation suggestions tailored to your location and space. "
        "Powered by GPT-4o for suggestions and DALL-E 3 for visualizations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(renovation_router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "House Renovation Advisor API",
        "docs": "/docs",
        "health": "/health",
    }
