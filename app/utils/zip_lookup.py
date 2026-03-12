import csv
from pathlib import Path
from functools import lru_cache
from typing import Optional
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

CROSSWALK_PATH = settings.DATA_DIR / "zip_county_crosswalk.csv"


@lru_cache()
def _load_crosswalk() -> dict[str, str]:
    """Load ZIP → county mapping. Returns empty dict if file not found."""
    if not CROSSWALK_PATH.exists():
        logger.warning(
            f"ZIP crosswalk not found at {CROSSWALK_PATH}. "
            "ZIP-based lookup will be unavailable."
        )
        return {}

    mapping: dict[str, str] = {}
    with open(CROSSWALK_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [c.lower() for c in (reader.fieldnames or [])]
        for row in reader:
            normalized = {k.lower(): v for k, v in row.items()}
            state = normalized.get("state", "").strip().upper()
            if state != "CA":
                continue
            zip_code = normalized.get("zip", "").strip().zfill(5)
            county = normalized.get("county", "").strip()
            if zip_code and county:
                mapping[zip_code] = county

    logger.info(f"Loaded {len(mapping)} CA ZIP → county mappings")
    return mapping


def zip_to_county(zip_code: str) -> Optional[str]:
    """Return county name for a California ZIP code."""
    mapping = _load_crosswalk()
    zip_code = zip_code.strip().zfill(5)
    county = mapping.get(zip_code)
    if county and not county.endswith("County"):
        county = f"{county} County"
    return county
