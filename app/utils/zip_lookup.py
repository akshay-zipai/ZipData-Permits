import csv
from pathlib import Path
from functools import lru_cache
from typing import Optional
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
from functools import lru_cache
from typing import Optional
from app.core.config import get_settings
from app.core.logging import get_logger
from pathlib import Path
import json

logger = get_logger(__name__)
settings = get_settings()

# Traditional CSV crosswalk (fallback)
CROSSWALK_PATH = settings.DATA_DIR / "zip_county_crosswalk.csv"

# JSON mapping produced by the build script (preferred if present)
MAPPING_JSON = settings.DATA_DIR / "california_permit_mapping.json"


@lru_cache()
def _load_crosswalk() -> dict[str, str]:
    """Load ZIP → county mapping from CSV fallback.
    Returns empty dict if file not found.
    """
    if not CROSSWALK_PATH.exists():
        logger.info(f"ZIP crosswalk not found at {CROSSWALK_PATH}. Using JSON mapping if available.")
        return {}

    mapping: dict[str, str] = {}
    with open(CROSSWALK_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {k.lower(): (v or "") for k, v in row.items()}
            state = normalized.get("state", "").strip().upper()
            if state != "CA":
                continue
            zip_code = normalized.get("zip", "").strip().zfill(5)
            county = normalized.get("county", "").strip()
            if zip_code and county:
                mapping[zip_code] = county

    logger.info(f"Loaded {len(mapping)} CA ZIP → county mappings from CSV")
    return mapping


@lru_cache()
def _load_mapping_json() -> dict[str, str]:
    """Load mapping produced in data/california_permit_mapping.json.
    Returns mapping zip -> county if available, else empty dict.
    """
    if not MAPPING_JSON.exists():
        return {}
    try:
        data = json.loads(MAPPING_JSON.read_text(encoding="utf-8"))
    except Exception:
        logger.warning(f"Failed to read JSON mapping at {MAPPING_JSON}")
        return {}

    out: dict[str, str] = {}
    for rec in data:
        county = rec.get("county_name")
        zips = rec.get("zip_codes") or []
        if county and zips:
            for z in zips:
                out[str(z).zfill(5)] = county
    logger.info(f"Loaded {len(out)} ZIP → county mappings from JSON mapping")
    return out


def zip_to_county(zip_code: str) -> Optional[str]:
    """Return county name for a California ZIP code.

    Prefer the JSON mapping in `data/` if present; otherwise fall back
    to the CSV crosswalk.
    """
    z = zip_code.strip().zfill(5)
    mapping_json = _load_mapping_json()
    county = mapping_json.get(z)
    if not county:
        mapping_csv = _load_crosswalk()
        county = mapping_csv.get(z)

    if county and not county.endswith("County"):
        county = f"{county} County"
    return county
