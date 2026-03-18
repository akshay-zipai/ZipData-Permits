"""
scripts/download_crosswalk.py
Downloads the Census ZCTA→County relationship file and generates
data/zip_county_crosswalk.csv for California ZIPs.

Run standalone:  python scripts/download_crosswalk.py
Auto-run by:     scripts/entrypoint.sh on container start
"""
import csv
import sys
import requests
from pathlib import Path
from io import StringIO

CENSUS_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/"
    "tab20_zcta520_county20_natl.txt"
)
OUTPUT_PATH = Path("data/zip_county_crosswalk.csv")
CA_FIPS_PREFIX = "06"  # California state FIPS code


def download_crosswalk() -> None:
    Path("data").mkdir(parents=True, exist_ok=True)

    print(f"Downloading Census ZCTA→County crosswalk from:\n  {CENSUS_URL}")
    try:
        resp = requests.get(CENSUS_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Failed to download crosswalk: {e}", file=sys.stderr)
        sys.exit(1)

    reader = csv.DictReader(StringIO(resp.text), delimiter="|")
    rows: dict[str, dict[str, str]] = {}

    for row in reader:
        county_fips = (row.get("GEOID_COUNTY_20") or "").strip()
        if not county_fips.startswith(CA_FIPS_PREFIX):
            continue

        zip_code = (row.get("GEOID_ZCTA5_20") or "").strip().zfill(5)
        county = (row.get("NAMELSAD_COUNTY_20") or "").strip()
        if zip_code and county and zip_code not in rows:
            rows[zip_code] = {"zip": zip_code, "county": county, "state": "CA"}

    if not rows:
        print("ERROR: No California rows found in crosswalk.", file=sys.stderr)
        sys.exit(1)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["zip", "county", "state"])
        writer.writeheader()
        for zip_code in sorted(rows):
            writer.writerow(rows[zip_code])

    print(f"Saved {len(rows)} CA ZIP codes to {OUTPUT_PATH}")


if __name__ == "__main__":
    download_crosswalk()
