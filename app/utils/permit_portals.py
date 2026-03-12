import json
from pathlib import Path
from functools import lru_cache
from typing import Optional
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache()
def load_permit_portals() -> dict[str, str]:
    """Load county → portal URL mapping from JSON file."""
    path: Path = settings.PERMIT_PORTALS_PATH
    if not path.exists():
        logger.warning(f"Permit portals file not found: {path}")
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} permit portals from {path}")
    return data


def get_portal_url(county_name: str) -> Optional[str]:
    """Return the permit portal URL for a given county name."""
    portals = load_permit_portals()
    # Try exact match first
    if county_name in portals:
        return portals[county_name]
    # Try with 'County' suffix
    if not county_name.endswith("County"):
        key = f"{county_name} County"
        if key in portals:
            return portals[key]
    # Case-insensitive fallback
    lower = county_name.lower()
    for k, v in portals.items():
        if k.lower() == lower or k.lower() == f"{lower} county":
            return v
    return None


def get_all_portals() -> dict[str, str]:
    return load_permit_portals()


def list_counties() -> list[str]:
    return sorted(load_permit_portals().keys())
