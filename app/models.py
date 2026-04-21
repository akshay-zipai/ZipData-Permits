from pydantic import BaseModel, Field
from typing import List, Optional


# ── Request ──────────────────────────────────────────────────────────────────

class RenovationRequest(BaseModel):
    place: str = Field(..., description="City, region, or country (e.g. 'Jaipur, India')", min_length=2)
    house_part: str = Field(..., description="Part of the house to renovate (e.g. 'kitchen', 'living room')", min_length=2)
    query: Optional[str] = Field(default="", description="Any additional preferences or requirements")
    generate_image: bool = Field(default=True, description="Whether to generate a visualization image")


# ── Response ─────────────────────────────────────────────────────────────────

class RenovationSuggestion(BaseModel):
    title: str
    description: str
    style: str
    budget_tier: str
    key_materials: List[str]
    estimated_duration: str
    pros: List[str]
    local_tip: str


class RenovationResponse(BaseModel):
    place: str
    house_part: str
    summary: str
    suggestions: List[RenovationSuggestion]
    image_url: Optional[str] = None
    image_prompt: Optional[str] = None


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    service: str
