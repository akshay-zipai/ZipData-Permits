import logging
from fastapi import APIRouter, HTTPException

from app.models import RenovationRequest, RenovationResponse, RenovationSuggestion
from app.services import get_renovation_suggestions, generate_renovation_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/renovation", tags=["Renovation"])


@router.post("/suggest", response_model=RenovationResponse)
async def suggest_renovation(request: RenovationRequest):
    """
    Get AI-powered renovation suggestions for a specific part of your house.

    - **place**: Your location (e.g. 'Jaipur, India', 'Miami, Florida')
    - **house_part**: Area to renovate (e.g. 'kitchen', 'bathroom', 'living room', 'facade')
    - **query**: Any additional preferences (budget, style, materials, etc.)
    - **generate_image**: Set to true to also get a DALL-E visualization
    """
    try:
        # Step 1: Get LLM suggestions
        data = await get_renovation_suggestions(
            place=request.place,
            house_part=request.house_part,
            query=request.query or "",
        )

        # Step 2: Parse suggestions
        suggestions = [RenovationSuggestion(**s) for s in data.get("suggestions", [])]

        # Step 3: Optionally generate image
        image_url = None
        image_prompt = data.get("image_prompt")

        if request.generate_image and image_prompt:
            try:
                image_url = await generate_renovation_image(image_prompt)
            except Exception as img_err:
                logger.warning(f"Image generation failed (non-fatal): {img_err}")

        return RenovationResponse(
            place=data.get("place", request.place),
            house_part=data.get("house_part", request.house_part),
            summary=data.get("summary", ""),
            suggestions=suggestions,
            image_url=image_url,
            image_prompt=image_prompt,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid response from LLM: {e}")
    except Exception as e:
        logger.error(f"Renovation suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
