from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from agent.config import get_settings

settings = get_settings()


class RenovationCollageStore:
    def __init__(self) -> None:
        region = settings.S3_REGION or settings.BEDROCK_REGION
        kwargs: dict[str, Any] = {"region_name": region}
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
            if settings.AWS_SESSION_TOKEN:
                kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN
        self._s3 = boto3.client("s3", **kwargs)

    @property
    def enabled(self) -> bool:
        return bool(settings.S3_RENOVATION_BUCKET)

    def build_cache_key(
        self,
        place: str,
        house_part: str,
        user_prefs: str,
        max_suggestions: int,
    ) -> str:
        normalized = {
            "place": (place or "").strip().lower(),
            "house_part": (house_part or "").strip().lower(),
            "user_prefs": " ".join((user_prefs or "").strip().lower().split()),
            "max_suggestions": max_suggestions,
        }
        raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_cached_collage(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        index = self._load_index()
        record = index.get(cache_key)
        if not record:
            return None

        try:
            image_obj = self._s3.get_object(
                Bucket=settings.S3_RENOVATION_BUCKET,
                Key=record["collage_s3_key"],
            )
            image_bytes = image_obj["Body"].read()
        except ClientError as exc:
            print(f"[S3Cache] Failed to read cached collage: {exc}")
            return None

        return {
            "image_bytes": image_bytes,
            "collage_s3_key": record["collage_s3_key"],
            "metadata_s3_key": record["metadata_s3_key"],
            "metadata_url": self.presign_url(record["metadata_s3_key"]),
            "collage_url": self.presign_url(record["collage_s3_key"]),
            "cached_at": record.get("created_at"),
        }

    def put_collage(
        self,
        *,
        cache_key: str,
        image_bytes: bytes,
        metadata: Dict[str, Any],
    ) -> Dict[str, str]:
        if not self.enabled:
            raise RuntimeError("S3 renovation bucket is not configured.")

        timestamp = datetime.now(timezone.utc)
        stamp = timestamp.strftime("%Y/%m/%d/%H%M%S")
        short_key = cache_key[:12]
        base_key = f"{settings.S3_RENOVATION_PREFIX}/{stamp}-{short_key}"
        collage_s3_key = f"{base_key}/collage.png"
        metadata_s3_key = f"{base_key}/metadata.json"

        self._s3.put_object(
            Bucket=settings.S3_RENOVATION_BUCKET,
            Key=collage_s3_key,
            Body=image_bytes,
            ContentType="image/png",
        )
        self._s3.put_object(
            Bucket=settings.S3_RENOVATION_BUCKET,
            Key=metadata_s3_key,
            Body=json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )

        index = self._load_index()
        index[cache_key] = {
            "cache_key": cache_key,
            "collage_s3_key": collage_s3_key,
            "metadata_s3_key": metadata_s3_key,
            "created_at": timestamp.isoformat(),
            "place": metadata.get("place"),
            "house_part": metadata.get("house_part"),
            "styles": metadata.get("styles", []),
            "budget_tiers": metadata.get("budget_tiers", []),
        }
        self._save_index(index)

        return {
            "collage_s3_key": collage_s3_key,
            "metadata_s3_key": metadata_s3_key,
            "collage_url": self.presign_url(collage_s3_key),
            "metadata_url": self.presign_url(metadata_s3_key),
        }

    def presign_url(self, key: str) -> str:
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_RENOVATION_BUCKET, "Key": key},
            ExpiresIn=settings.S3_URL_EXPIRY_SECONDS,
        )

    def _load_index(self) -> Dict[str, Any]:
        try:
            obj = self._s3.get_object(
                Bucket=settings.S3_RENOVATION_BUCKET,
                Key=settings.S3_RENOVATION_INDEX_KEY,
            )
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in {"NoSuchKey", "404"}:
                return {}
            raise

        body = obj["Body"].read().decode("utf-8").strip()
        if not body:
            return {}
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            print("[S3Cache] Index JSON is malformed. Starting with empty index.")
            return {}

        if isinstance(payload, dict):
            return payload
        return {}

    def _save_index(self, index: Dict[str, Any]) -> None:
        self._s3.put_object(
            Bucket=settings.S3_RENOVATION_BUCKET,
            Key=settings.S3_RENOVATION_INDEX_KEY,
            Body=json.dumps(index, indent=2, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )

    @staticmethod
    def data_uri_to_bytes(data_uri: str) -> bytes:
        _, b64_data = data_uri.split(",", 1)
        return base64.b64decode(b64_data)

    @staticmethod
    def bytes_to_data_uri(image_bytes: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
