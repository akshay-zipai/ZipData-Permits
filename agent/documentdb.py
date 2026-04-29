from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, Field

from agent.config import Settings, get_settings


class LocationSnapshot(BaseModel):
    state_code: str = "CA"
    zip_code: Optional[str] = None
    county_name: Optional[str] = None
    city: Optional[str] = None


class RetrievalSnapshot(BaseModel):
    top_k: int
    chunks_used: int
    county_filter: Optional[str] = None
    zip_filter: Optional[str] = None
    source_urls: list[str] = Field(default_factory=list)


class AnswerSnapshot(BaseModel):
    text: str
    generated_at: str
    model_provider: str
    model_name: str


class QuestionWorkflowSnapshot(BaseModel):
    state_before: str
    state_after: str
    permit_count: int
    reno_count: int


class DatasetQuestionDocument(BaseModel):
    question_id: str
    session_id: str
    question_type: str = "permit"
    question_text: str
    normalized_question: str
    question_hash: str
    location: LocationSnapshot
    retrieval: RetrievalSnapshot
    answer: AnswerSnapshot
    workflow: QuestionWorkflowSnapshot
    tags: list[str] = Field(default_factory=list)
    app_name: str
    app_version: str
    environment: str
    created_at: str
    updated_at: str


class ConversationSessionDocument(BaseModel):
    session_id: str
    status: str
    started_at: str
    last_seen_at: str
    ended_at: Optional[str] = None
    app_name: str
    app_version: str
    environment: str


class DocumentDBQuestionStore:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.enabled = self.settings.DOCUMENTDB_ENABLED
        self._client = None
        self._db = None
        self._questions = None
        self._sessions = None

    @property
    def is_ready(self) -> bool:
        return bool(self._questions is not None and self._sessions is not None)

    def initialize(self) -> None:
        if not self.enabled:
            return

        client = self._build_client()
        db = client[self.settings.DOCUMENTDB_DATABASE]
        questions = db[self.settings.DOCUMENTDB_QUESTIONS_COLLECTION]
        sessions = db[self.settings.DOCUMENTDB_SESSIONS_COLLECTION]

        questions.create_index("question_id", unique=True, name="uq_question_id")
        questions.create_index(
            [("created_at", -1), ("question_type", 1)],
            name="idx_questions_created_type",
        )
        questions.create_index(
            [("location.county_name", 1), ("location.zip_code", 1), ("created_at", -1)],
            name="idx_questions_location_time",
        )
        questions.create_index(
            [("question_hash", 1), ("location.county_name", 1)],
            name="idx_questions_hash_county",
        )
        questions.create_index(
            [("session_id", 1), ("created_at", -1)],
            name="idx_questions_session_time",
        )

        sessions.create_index("session_id", unique=True, name="uq_session_id")
        sessions.create_index(
            [("status", 1), ("last_seen_at", -1)],
            name="idx_sessions_status_last_seen",
        )

        client.admin.command("ping")

        self._client = client
        self._db = db
        self._questions = questions
        self._sessions = sessions

    def upsert_session_start(self, session_id: str) -> None:
        if not self.is_ready:
            return

        now = _utcnow()
        document = ConversationSessionDocument(
            session_id=session_id,
            status="active",
            started_at=now,
            last_seen_at=now,
            app_name=self.settings.APP_NAME,
            app_version=self.settings.APP_VERSION,
            environment=self.settings.ENVIRONMENT,
        )
        self._sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": document.status,
                    "last_seen_at": document.last_seen_at,
                    "app_name": document.app_name,
                    "app_version": document.app_version,
                    "environment": document.environment,
                },
                "$setOnInsert": {
                    "session_id": document.session_id,
                    "started_at": document.started_at,
                },
                "$unset": {"ended_at": ""},
            },
            upsert=True,
        )

    def close_session(self, session_id: str, status: str = "completed") -> None:
        if not self.is_ready:
            return

        now = _utcnow()
        self._sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": status,
                    "last_seen_at": now,
                    "ended_at": now,
                },
                "$setOnInsert": {
                    "session_id": session_id,
                    "started_at": now,
                    "app_name": self.settings.APP_NAME,
                    "app_version": self.settings.APP_VERSION,
                    "environment": self.settings.ENVIRONMENT,
                },
            },
            upsert=True,
        )

    def store_permit_question(
        self,
        *,
        session_id: str,
        question_text: str,
        county_name: Optional[str],
        zip_code: Optional[str],
        city: Optional[str],
        chunks: list[Any],
        answer: str,
        state_before: str,
        state_after: str,
        permit_count: int,
        reno_count: int,
    ) -> str:
        if not self.is_ready:
            return ""

        now = _utcnow()
        normalized_question = _normalize_question(question_text)
        sources = list(dict.fromkeys(chunk.source_url for chunk in chunks if chunk.source_url))
        document = DatasetQuestionDocument(
            question_id=str(uuid.uuid4()),
            session_id=session_id,
            question_text=question_text.strip(),
            normalized_question=normalized_question,
            question_hash=_question_hash(normalized_question, county_name, zip_code),
            location=LocationSnapshot(
                zip_code=zip_code,
                county_name=county_name,
                city=city,
            ),
            retrieval=RetrievalSnapshot(
                top_k=self.settings.RAG_TOP_K,
                chunks_used=len(chunks),
                county_filter=county_name,
                zip_filter=zip_code,
                source_urls=sources,
            ),
            answer=AnswerSnapshot(
                text=answer,
                generated_at=now,
                model_provider="bedrock" if self.settings.is_production else "openai",
                model_name=(
                    self.settings.BEDROCK_MODEL_ID
                    if self.settings.is_production
                    else self.settings.OPENAI_MODEL
                ),
            ),
            workflow=QuestionWorkflowSnapshot(
                state_before=state_before,
                state_after=state_after,
                permit_count=permit_count,
                reno_count=reno_count,
            ),
            tags=_build_tags(question_text),
            app_name=self.settings.APP_NAME,
            app_version=self.settings.APP_VERSION,
            environment=self.settings.ENVIRONMENT,
            created_at=now,
            updated_at=now,
        )

        self._questions.insert_one(document.model_dump(mode="json"))
        self._sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {"last_seen_at": now, "status": "active"},
                "$setOnInsert": {
                    "session_id": session_id,
                    "started_at": now,
                    "app_name": self.settings.APP_NAME,
                    "app_version": self.settings.APP_VERSION,
                    "environment": self.settings.ENVIRONMENT,
                },
            },
            upsert=True,
        )
        return document.question_id

    def _build_client(self):
        from pymongo import MongoClient

        uri = self.settings.DOCUMENTDB_URI
        if not uri:
            host = self.settings.DOCUMENTDB_HOST or self._resolve_cluster_endpoint()
            if not host:
                raise ValueError(
                    "DocumentDB is enabled but no host or resolvable cluster endpoint is configured."
                )
            uri = self.settings.build_documentdb_uri(host)

        client_options: dict[str, Any] = {
            "appname": self.settings.APP_NAME,
            "connectTimeoutMS": self.settings.DOCUMENTDB_CONNECT_TIMEOUT_MS,
            "serverSelectionTimeoutMS": self.settings.DOCUMENTDB_SERVER_SELECTION_TIMEOUT_MS,
        }
        if self.settings.DOCUMENTDB_TLS_CA_FILE:
            client_options["tlsCAFile"] = self.settings.DOCUMENTDB_TLS_CA_FILE

        return MongoClient(uri, **client_options)

    def _resolve_cluster_endpoint(self) -> Optional[str]:
        if not self.settings.DOCUMENTDB_CLUSTER_ID:
            return None

        region = self.settings.BEDROCK_REGION
        kwargs: dict[str, Any] = {"region_name": region}
        if self.settings.AWS_ACCESS_KEY_ID and self.settings.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = self.settings.AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = self.settings.AWS_SECRET_ACCESS_KEY
            if self.settings.AWS_SESSION_TOKEN:
                kwargs["aws_session_token"] = self.settings.AWS_SESSION_TOKEN

        try:
            client = boto3.client("docdb", **kwargs)
            response = client.describe_db_clusters(
                DBClusterIdentifier=self.settings.DOCUMENTDB_CLUSTER_ID
            )
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Unable to resolve DocumentDB cluster endpoint: {exc}") from exc

        clusters = response.get("DBClusters", [])
        if not clusters:
            raise RuntimeError("DocumentDB cluster lookup returned no clusters.")
        return clusters[0].get("Endpoint")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_question(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _question_hash(normalized_question: str, county_name: Optional[str], zip_code: Optional[str]) -> str:
    payload = "|".join([normalized_question, county_name or "", zip_code or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_tags(question_text: str) -> list[str]:
    keywords = {
        "adu": "adu",
        "kitchen": "kitchen",
        "bathroom": "bathroom",
        "solar": "solar",
        "deck": "deck",
        "garage": "garage",
        "fee": "fees",
        "cost": "costs",
        "timeline": "timeline",
        "how long": "timeline",
        "inspection": "inspection",
        "remodel": "remodel",
    }
    lowered = (question_text or "").lower()
    tags = {tag for term, tag in keywords.items() if term in lowered}
    return sorted(tags)


_store: Optional[DocumentDBQuestionStore] = None


def get_question_store() -> DocumentDBQuestionStore:
    global _store
    if _store is None:
        _store = DocumentDBQuestionStore()
    return _store
