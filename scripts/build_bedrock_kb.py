"""
Build a Bedrock-ready knowledge base JSONL from the offline dataset.

This script reads an existing JSONL (the output of the crawler/embedding
pipeline) and writes a simplified JSONL suitable for uploading to S3 and
consumption by the RAG pipeline + Bedrock LLM. By default embeddings are
removed and only essential metadata retained.

Output schema (one object per line):
  {
    "id": "uuid",
    "chunk_id": "source_url#idx",
    "county_name": "Alameda County",
    "zip_codes": ["94501","94502"] | null,
    "source_url": "https://...",
    "title": "Page title",
    "content": "cleaned chunk text",
    "word_count": 123
  }

Usage:
  python scripts/build_bedrock_kb.py --in output/offline_dataset_full.jsonl --out output/bedrock_kb.jsonl --zip-list

If `--zip-list` is set and `data/zip_county_crosswalk.csv` exists, the script
will attach a list of ZIP codes for each county (useful for ZIP-based filtering).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path as _Path
import json as _json


def load_mapping_by_county(mapping_path: _Path) -> Dict[str, List[str]]:
    """Load county -> zip_codes mapping from data/california_permit_mapping.json.
    Returns dict county_name -> list of zips.
    """
    if not mapping_path.exists():
        return {}
    try:
        data = _json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, List[str]] = {}
    for rec in data:
        county = rec.get("county_name")
        zips = rec.get("zip_codes") or []
        if county:
            out[county] = zips
    return out


def build_bedrock_kb(in_path: Path, out_path: Path, include_zip_list: bool = False) -> int:
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")

    # Prefer the generated mapping in data/california_permit_mapping.json
    mapping_file = Path("data/california_permit_mapping.json")
    if include_zip_list:
        zip_by_county = load_mapping_by_county(mapping_file)
    else:
        zip_by_county = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with in_path.open("r", encoding="utf-8") as fh_in, out_path.open("w", encoding="utf-8") as fh_out:
        for line in fh_in:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            chunk_id = rec.get("chunk_id") or rec.get("id")
            county = rec.get("county_name")
            content = rec.get("content") or rec.get("text") or rec.get("text_content") or ""
            title = rec.get("title")
            source_url = rec.get("source_url")
            word_count = rec.get("word_count") or (len(content.split()))

            if include_zip_list and county and county in zip_by_county and zip_by_county[county]:
                # Expand one output record per ZIP code
                for z in zip_by_county[county]:
                    out_rec = {
                        "id": f"{chunk_id}::{z}",
                        "chunk_id": chunk_id,
                        "zip_code": z,
                        "county_name": county,
                        "source_url": source_url,
                        "title": title,
                        "content": content,
                        "word_count": word_count,
                    }
                    fh_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    count += 1
            else:
                out_rec = {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "zip_code": None,
                    "county_name": county,
                    "source_url": source_url,
                    "title": title,
                    "content": content,
                    "word_count": word_count,
                }
                fh_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                count += 1

    # write simple manifest for upload convenience
    manifest = {"version": 1, "files": [{"path": str(out_path), "format": "jsonl", "has_embeddings": False}]}
    manifest_path = out_path.parent / "manifest_bedrock.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return count


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Build Bedrock KB JSONL from offline dataset")
    parser.add_argument("--in", dest="infile", required=True, help="Input offline JSONL path")
    parser.add_argument("--out", dest="outfile", required=True, help="Output JSONL path for Bedrock KB")
    parser.add_argument("--zip-list", action="store_true", help="Attach list of ZIP codes per county if crosswalk exists")

    args = parser.parse_args(argv)
    in_path = Path(args.infile)
    out_path = Path(args.outfile)

    total = build_bedrock_kb(in_path, out_path, include_zip_list=args.zip_list)
    print(f"Wrote {total} records to {out_path}")
    print(f"Manifest written to {out_path.parent / 'manifest_bedrock.json'}")


if __name__ == "__main__":
    main()
