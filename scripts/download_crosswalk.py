"""
scripts/download_crosswalk.py
Downloads the Census ZCTA→County relationship file and generates
data/zip_county_crosswalk.csv for California ZIPs.

Run standalone:  python scripts/download_crosswalk.py
Auto-run by:     scripts/entrypoint.sh on container start
"""
import sys
import requests
import pandas as pd
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

    df = pd.read_csv(StringIO(resp.text), sep="|", dtype=str)

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Keep only California rows (FIPS county code starts with "06")
    ca_mask = df["GEOID_COUNTY_20"].str[:2] == CA_FIPS_PREFIX
    df = df[ca_mask].copy()

    if df.empty:
        print("ERROR: No California rows found in crosswalk.", file=sys.stderr)
        sys.exit(1)

    # Build output dataframe
    out = pd.DataFrame()
    out["zip"]    = df["GEOID_ZCTA5_20"].str.zfill(5)
    out["county"] = df["NAMELSAD_COUNTY_20"].str.strip()
    out["state"]  = "CA"

    # Remove duplicates (a ZIP can span multiple counties — keep first/largest)
    out = out.drop_duplicates(subset=["zip"]).sort_values("zip").reset_index(drop=True)

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out)} CA ZIP codes to {OUTPUT_PATH}")


if __name__ == "__main__":
    download_crosswalk()
