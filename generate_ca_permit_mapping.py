"""
generate_ca_permit_mapping.py — Standalone script (no FastAPI dependency).
Reads the ZIP crosswalk + permit portals JSON and produces a combined mapping.

Usage:
    python generate_ca_permit_mapping.py

Outputs:
    output/california_permit_mapping.json
"""
import csv
import json
import os
from pathlib import Path

CROSSWALK_PATH = Path("data/zip_county_crosswalk.csv")
PERMIT_PORTAL_PATH = Path("data/permit_portals.json")
OUTPUT_PATH = Path(os.getenv("PERMIT_MAPPING_OUTPUT_PATH", "output/california_permit_mapping.json"))
STATE_FILTER = "CA"


def load_crosswalk(path: Path) -> dict[str, list[str]]:
    """Return county_name → [zip_codes] mapping for California."""
    if not path.exists():
        raise FileNotFoundError(f"Crosswalk file not found: {path}")

    county_zips: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {k.lower(): v for k, v in row.items()}
            if normalized.get("state", "").strip().upper() != STATE_FILTER:
                continue
            zip_code = normalized.get("zip", "").strip().zfill(5)
            county = normalized.get("county", "").strip()
            if not county.endswith("County"):
                county = f"{county} County"
            county_zips.setdefault(county, [])
            if zip_code not in county_zips[county]:
                county_zips[county].append(zip_code)

    return {k: sorted(v) for k, v in county_zips.items()}


def load_permit_portals(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_mapping(
    county_zips: dict[str, list[str]],
    permit_portals: dict[str, str],
) -> list[dict]:
    results = []
    for county_name, zips in sorted(county_zips.items()):
        results.append({
            "county_name": county_name,
            "zip_codes": zips,
            "official_permit_portal": permit_portals.get(county_name, ""),
        })
    return results


def save_output(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} counties to {path}")


if __name__ == "__main__":
    print("Loading crosswalk...")
    county_zips = load_crosswalk(CROSSWALK_PATH)

    print("Loading permit portals...")
    portals = load_permit_portals(PERMIT_PORTAL_PATH)

    print("Generating mapping...")
    mapping = generate_mapping(county_zips, portals)

    print("Saving output...")
    save_output(mapping, OUTPUT_PATH)
    print("Done.")
