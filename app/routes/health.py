from fastapi import APIRouter
from app.models import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", service="house-renovation-api")
