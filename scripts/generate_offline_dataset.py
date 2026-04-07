"""
Generate offline dataset from permit portals.

Produces a newline-delimited JSON file (`.jsonl`) containing chunks with
embeddings ready for use by a Bedrock-backed RAG workflow or other vector
retrieval systems. Optionally uploads the generated files to S3.

Usage examples:
  python scripts/generate_offline_dataset.py --counties "Los Angeles" "Alameda" --out output/offline.jsonl --upload --s3-bucket my-bucket

The output JSONL schema (one object per line):
  {
    "id": "uuid",
    "chunk_id": "source_url#idx",
    "county_name": "Los Angeles",
    "source_url": "https://...",
    "title": "Page Title",
    "content": "cleaned chunk text",
    "word_count": 123,
    "embedding": [0.123, ...]
  }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Iterable, List

# Ensure project root is on `sys.path` so `app` package imports work when
# running this script directly (e.g. `python scripts/generate_offline_dataset.py`).
# This is safe and makes local runs simple; alternatively set `PYTHONPATH`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.crawling.crawler import get_crawler_service
from app.services.embedding.embedder import get_embedding_service
from app.utils.permit_portals import get_portal_url, list_counties
from app.utils.text_processing import chunk_text, clean_text
import pandas as pd


def write_jsonl(path: Path, records: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def upload_to_s3(file_path: Path, bucket: str, key: str):
    import boto3

    s3 = boto3.client("s3")
    s3.upload_file(str(file_path), bucket, key)


async def generate_for_counties(
    counties: List[str], out_path: Path, batch_size: int = 64
):
    crawler = get_crawler_service()
    embedder = get_embedding_service()

    # Remove output if exists to allow appending fresh results
    if out_path.exists():
        out_path.unlink()

    parquet_rows: List[dict] = []

    for county in counties:
        portal = get_portal_url(county)
        if not portal:
            print(f"No portal URL for county: {county}")
            continue

        print(f"Crawling {county}: {portal}")
        try:
            crawl_result = await crawler.crawl_url(portal, county, force_refresh=False)
        except Exception as e:
            print(f"Crawl failed for {county}: {e}")
            continue

        texts_batch: List[str] = []
        pending_records: List[dict] = []

        def flush_batch():
            if not texts_batch:
                return
            try:
                emb_resp = embedder.embed_texts(texts_batch)
                embeddings = emb_resp.embeddings
            except Exception as e:
                print(f"Embedding failed: {e}\nFalling back to empty embeddings for this batch.")
                embeddings = [[] for _ in texts_batch]
            for rec, emb in zip(pending_records, embeddings):
                rec["embedding"] = emb
            write_jsonl(out_path, pending_records)
            # store rows for parquet export (embedding kept as list)
            for r in pending_records:
                parquet_rows.append(r.copy())
            texts_batch.clear()
            pending_records.clear()

        for page in crawl_result.pages:
            if not page.text_content:
                continue
            title = page.title or ""
            chunks = chunk_text(clean_text(page.text_content))
            for idx, chunk in enumerate(chunks):
                rec = {
                    "id": str(uuid.uuid4()),
                    "chunk_id": f"{page.url}#{idx}",
                    "county_name": county,
                    "source_url": page.url,
                    "title": title,
                    "content": chunk,
                    "word_count": len(chunk.split()),
                }
                texts_batch.append(chunk)
                pending_records.append(rec)

                if len(texts_batch) >= batch_size:
                    flush_batch()

        # flush remaining
        flush_batch()

    # Write Parquet file if any rows collected
    parquet_path = out_path.with_suffix(".parquet")
    if parquet_rows:
        try:
            df = pd.DataFrame(parquet_rows)
            df.to_parquet(parquet_path, index=False)
            print(f"Parquet dataset written to {parquet_path}")
        except Exception as e:
            print(f"Parquet export failed: {e}")

    print(f"Offline dataset written to {out_path}")

    # Write local manifest.json referencing produced files
    manifest = {
        "version": 1,
        "files": [
            {"path": str(out_path), "format": "jsonl", "has_embeddings": True},
        ],
    }
    if parquet_rows:
        manifest["files"].append(
            {"path": str(parquet_path), "format": "parquet", "has_embeddings": True}
        )

    manifest_path = out_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Local manifest written to {manifest_path}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Generate offline permit dataset")
    parser.add_argument(
        "--counties",
        nargs="*",
        help="List of county names to crawl. If omitted, all portals from data file will be used.",
    )
    parser.add_argument("--out", default="output/offline_dataset.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--upload", action="store_true", help="Upload output to S3")
    parser.add_argument("--s3-bucket", help="S3 bucket name when --upload is set")
    parser.add_argument("--s3-key", help="S3 key (path) to upload the file to")

    args = parser.parse_args(argv)
    out_path = Path(args.out)

    if args.counties:
        counties = args.counties
    else:
        try:
            counties = list_counties()
        except Exception:
            print("Could not load portal list; please pass --counties")
            sys.exit(1)

    asyncio.run(generate_for_counties(counties, out_path, batch_size=args.batch_size))

    # If upload requested, upload JSONL, Parquet (if present) and manifest to S3
    if args.upload:
        if not args.s3_bucket:
            print("--s3-bucket is required when --upload is set")
            sys.exit(1)

        # Determine S3 keys
        s3_key_jsonl = args.s3_key or out_path.name
        s3_path = Path(s3_key_jsonl)
        s3_key_parquet = str(s3_path.with_suffix(".parquet"))
        s3_key_manifest = str(s3_path.with_name("manifest.json"))

        # Upload files
        print(f"Uploading {out_path} to s3://{args.s3_bucket}/{s3_key_jsonl}")
        upload_to_s3(out_path, args.s3_bucket, s3_key_jsonl)

        parquet_path = out_path.with_suffix(".parquet")
        if parquet_path.exists():
            print(f"Uploading {parquet_path} to s3://{args.s3_bucket}/{s3_key_parquet}")
            upload_to_s3(parquet_path, args.s3_bucket, s3_key_parquet)

        manifest_path = out_path.parent / "manifest.json"
        if manifest_path.exists():
            print(f"Uploading {manifest_path} to s3://{args.s3_bucket}/{s3_key_manifest}")
            upload_to_s3(manifest_path, args.s3_bucket, s3_key_manifest)

        # Create and upload an S3 manifest that references the uploaded S3 URIs
        s3_manifest = {
            "version": 1,
            "files": [],
        }
        s3_manifest["files"].append(
            {"s3_uri": f"s3://{args.s3_bucket}/{s3_key_jsonl}", "format": "jsonl", "has_embeddings": True}
        )
        if parquet_path.exists():
            s3_manifest["files"].append(
                {"s3_uri": f"s3://{args.s3_bucket}/{s3_key_parquet}", "format": "parquet", "has_embeddings": True}
            )

        s3_manifest_local = out_path.parent / "manifest_s3.json"
        s3_manifest_local.write_text(json.dumps(s3_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        s3_manifest_key = str(Path(s3_key_manifest).with_name("manifest_s3.json"))
        print(f"Uploading S3 manifest to s3://{args.s3_bucket}/{s3_manifest_key}")
        upload_to_s3(s3_manifest_local, args.s3_bucket, s3_manifest_key)

        print("Upload complete")


if __name__ == "__main__":
    main()
