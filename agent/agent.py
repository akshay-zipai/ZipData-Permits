"""
Conversational Agent — State machine guiding users through:
  1. Permit Q&A  (location → county/zip → question → RAG answer)
  2. Renovation  (area → preferences → AI suggestions + 4 images from one collage)

LLM calls go through agent.llm.get_llm_client() which transparently
dispatches to OpenAI (local) or AWS Bedrock (production).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from PIL import Image

from agent.config import get_settings
from agent.documentdb import get_question_store
from agent.llm import get_llm_client
from agent.permit_kb import get_permit_kb, PermitChunk
from agent.storage import RenovationCollageStore

settings = get_settings()


# ── States ────────────────────────────────────────────────────────────────────

class AgentState(str, Enum):
    GREETING                = "greeting"
    COLLECT_LOCATION        = "collect_location"
    CONFIRM_COUNTY          = "confirm_county"
    COLLECT_PERMIT_QUESTION = "collect_permit_question"
    ANSWERING_PERMIT        = "answering_permit"
    PERMIT_FOLLOWUP         = "permit_followup"
    TRANSITION_TO_RENO      = "transition_to_reno"
    COLLECT_RENO_AREA       = "collect_reno_area"
    COLLECT_RENO_PREFS      = "collect_reno_prefs"
    GENERATING_RENO         = "generating_reno"
    RENO_FOLLOWUP           = "reno_followup"
    DONE                    = "done"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ConversationContext:
    session_id: Optional[str] = None
    state: AgentState = AgentState.COLLECT_LOCATION
    zip_code: Optional[str] = None
    county_name: Optional[str] = None
    city: Optional[str] = None
    permit_question: Optional[str] = None
    permit_answers: List[str] = field(default_factory=list)
    reno_area: Optional[str] = None
    reno_prefs: Optional[str] = None
    reno_result: Optional[Dict[str, Any]] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    permit_count: int = 0
    reno_count: int = 0


@dataclass
class AgentResponse:
    message: str
    state: AgentState
    data: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None


# ── Prompts ───────────────────────────────────────────────────────────────────

PERMIT_SYSTEM = """You are a helpful California building permit expert.
Use the provided context excerpts from official county permit databases and websites.
Be concise and practical. Cite the county the information comes from.
If the context lacks a specific answer, say so and suggest contacting the county directly.
Always include permit portal URLs when they appear in the context."""

RENO_SYSTEM = """You are an expert interior designer and renovation consultant specialising in California homes.
Provide clear, actionable, inspiring renovation suggestions tailored to the user's location.
Consider local climate, materials, California building codes, and current design trends.
Respond ONLY with valid JSON — no markdown fences, no preamble, no trailing text.
Keep each field concise to avoid truncation: description max 2 sentences, image_prompt max 1 sentence."""

RENO_USER_TEMPLATE = """
I need {max_suggestions} renovation suggestions for:
- Location: {place} (California)
- Area of House: {house_part}
- Additional Requirements: {user_query}

Respond ONLY with valid JSON. Keep ALL string values SHORT to avoid truncation.

{{
  "place": "{place}",
  "house_part": "{house_part}",
  "summary": "One sentence overview.",
  "suggestions": [
    {{
      "title": "Short title (max 6 words)",
      "description": "Max 2 sentences describing the idea.",
      "style": "One style label",
      "budget_tier": "Budget or Mid-range or Premium",
      "key_materials": ["mat1", "mat2", "mat3"],
      "estimated_duration": "e.g. 2-4 weeks",
      "estimated_cost": "e.g. $5,000-$15,000",
      "pros": ["pro1", "pro2"],
      "local_tip": "One sentence CA-specific tip.",
      "image_prompt": "Photorealistic {house_part} render, {place} California, [style], [key material], natural light"
    }}
  ]
}}
"""

IMAGE_SUFFIX = (
    ", photorealistic architectural render, professional interior design photography, "
    "high resolution, natural lighting, 4K quality, California home"
)


# ── JSON repair ───────────────────────────────────────────────────────────────

def _repair_truncated_json(raw: str) -> str:
    """
    Attempt to fix a JSON string that was cut off mid-generation.
    Strategy: find the last complete suggestion object and close the structure.
    """
    # Strip markdown fences
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    # Try parsing as-is first
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # Find the last complete suggestion by locating the last complete }
    # inside the suggestions array, then close everything properly.
    # Step 1: find the suggestions array opening
    suggestions_start = raw.find('"suggestions"')
    if suggestions_start == -1:
        raise ValueError("No suggestions array found in response")

    arr_open = raw.find("[", suggestions_start)
    if arr_open == -1:
        raise ValueError("No suggestions array bracket found")

    # Step 2: walk through and find complete objects by tracking brace depth
    depth = 0
    last_complete_obj_end = -1
    i = arr_open + 1

    while i < len(raw):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete_obj_end = i
        i += 1

    if last_complete_obj_end == -1:
        raise ValueError("No complete suggestion objects found")

    # Step 3: truncate to the last complete object and close the structure
    truncated = raw[: last_complete_obj_end + 1]

    # Find the outer object opening to determine what we need to close
    # We need to close: suggestions array ] and outer object }
    repaired = truncated + "\n  ]\n}"

    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        # Last resort: try wrapping what we have
        raise ValueError(f"Could not repair truncated JSON. Last error position near char {last_complete_obj_end}")


# ── Agent ─────────────────────────────────────────────────────────────────────

class PermitRenoAgent:

    def __init__(self):
        self.llm = get_llm_client()
        self.kb  = get_permit_kb()
        self.storage = RenovationCollageStore()
        self.question_store = get_question_store()

    # ── Main dispatch ─────────────────────────────────────────────────────────

    async def process(self, user_message: str, ctx: ConversationContext) -> AgentResponse:
        ctx.history.append({"role": "user", "content": user_message})

        if ctx.state == AgentState.GREETING:
            return await self._greet(user_message, ctx)
        elif ctx.state == AgentState.COLLECT_LOCATION:
            return await self._handle_location(user_message, ctx)
        elif ctx.state == AgentState.CONFIRM_COUNTY:
            return await self._handle_county_confirm(user_message, ctx)
        elif ctx.state == AgentState.COLLECT_PERMIT_QUESTION:
            return await self._handle_permit_question(user_message, ctx)
        elif ctx.state in (AgentState.ANSWERING_PERMIT, AgentState.PERMIT_FOLLOWUP):
            return await self._handle_permit_followup(user_message, ctx)
        elif ctx.state == AgentState.TRANSITION_TO_RENO:
            return await self._handle_reno_transition(user_message, ctx)
        elif ctx.state == AgentState.COLLECT_RENO_AREA:
            return await self._handle_reno_area(user_message, ctx)
        elif ctx.state == AgentState.COLLECT_RENO_PREFS:
            return await self._handle_reno_prefs(user_message, ctx)
        elif ctx.state == AgentState.RENO_FOLLOWUP:
            return await self._handle_reno_followup(user_message, ctx)
        else:
            return await self._fallback(user_message, ctx)

    def start(self) -> AgentResponse:
        return AgentResponse(
            message=(
                "👋 Welcome to the **California Permit & Renovation Agent**!\n\n"
                "I can help you with:\n"
                "1. 🏛️ **Building permit information** — county & ZIP-specific answers from offline CA data\n"
                "2. 🏠 **Renovation suggestions** — AI-powered design ideas with visualizations\n\n"
                "Let's start — **what's your California ZIP code or county?**"
            ),
            state=AgentState.COLLECT_LOCATION,
            suggestions=["Enter ZIP code", "Enter county name"],
        )

    # ── Location ──────────────────────────────────────────────────────────────

    async def _greet(self, msg: str, ctx: ConversationContext) -> AgentResponse:
        ctx.state = AgentState.COLLECT_LOCATION
        return AgentResponse(
            message="**What's your California ZIP code or county?**",
            state=ctx.state,
        )

    async def _handle_location(self, msg: str, ctx: ConversationContext) -> AgentResponse:
        zip_match = re.search(r"\b(9\d{4})\b", msg)
        if zip_match:
            zip_code = zip_match.group(1)
            county = self.kb.county_for_zip(zip_code)
            ctx.zip_code = zip_code
            if county:
                ctx.county_name = county
                ctx.state = AgentState.COLLECT_PERMIT_QUESTION
                return AgentResponse(
                    message=(
                        f"✅ Found **{county}** for ZIP {zip_code}.\n\n"
                        "**What's your building permit question?** For example:\n"
                        "- What permits do I need to add a room?\n"
                        "- How long does a deck permit take?\n"
                        "- What are the fees for a kitchen remodel?"
                    ),
                    state=ctx.state,
                    suggestions=[
                        "Permits for room addition",
                        "Kitchen remodel permit",
                        "ADU permit requirements",
                        "Solar panel permit",
                    ],
                )
            else:
                ctx.state = AgentState.CONFIRM_COUNTY
                return AgentResponse(
                    message=(
                        f"I couldn't find county data for ZIP {zip_code}. "
                        "Please enter your **county name** (e.g. 'Los Angeles County'):"
                    ),
                    state=ctx.state,
                    suggestions=self.kb.get_all_counties()[:6],
                )

        msg_lower = msg.lower()
        matched = next(
            (c for c in self.kb.get_all_counties()
             if c.lower() in msg_lower or msg_lower in c.lower()),
            None,
        )
        if matched:
            ctx.county_name = matched
            ctx.state = AgentState.COLLECT_PERMIT_QUESTION
            return AgentResponse(
                message=f"✅ Looking up permit info for **{matched}**.\n\n**What's your permit question?**",
                state=ctx.state,
                suggestions=["Room addition permit", "Kitchen remodel permit", "ADU permit", "Fence permit"],
            )

        ctx.state = AgentState.CONFIRM_COUNTY
        return AgentResponse(
            message=(
                "I didn't recognise that location. Please enter a **California ZIP code** "
                "or **county name** — e.g. '94501' or 'San Diego County'."
            ),
            state=ctx.state,
        )

    async def _handle_county_confirm(self, msg: str, ctx: ConversationContext) -> AgentResponse:
        zip_match = re.search(r"\b(9\d{4})\b", msg)
        if zip_match:
            ctx.zip_code = zip_match.group(1)
            county = self.kb.county_for_zip(ctx.zip_code)
            if county:
                ctx.county_name = county
                ctx.state = AgentState.COLLECT_PERMIT_QUESTION
                return AgentResponse(
                    message=f"✅ Found **{county}** for ZIP {ctx.zip_code}. What's your permit question?",
                    state=ctx.state,
                    suggestions=["Room addition permit", "ADU permit", "Solar permit"],
                )

        matched = next(
            (c for c in self.kb.get_all_counties()
             if c.lower() in msg.lower() or msg.lower().replace(" county", "") in c.lower()),
            None,
        )
        if matched:
            ctx.county_name = matched
            ctx.state = AgentState.COLLECT_PERMIT_QUESTION
            return AgentResponse(
                message=f"✅ Got it — **{matched}**. What's your permit question?",
                state=ctx.state,
                suggestions=["Room addition permit", "ADU permit", "Solar permit", "Deck permit"],
            )

        return AgentResponse(
            message="Still couldn't match a county. Please enter the full name, e.g. **'Los Angeles County'**.",
            state=ctx.state,
        )

    # ── Permit RAG ────────────────────────────────────────────────────────────

    async def _handle_permit_question(self, msg: str, ctx: ConversationContext) -> AgentResponse:
        ctx.permit_question = msg
        ctx.state = AgentState.ANSWERING_PERMIT
        return await self._answer_permit(ctx)

    async def _answer_permit(self, ctx: ConversationContext) -> AgentResponse:
        state_before = ctx.state.value
        chunks = self.kb.retrieve(
            query=ctx.permit_question,
            top_k=settings.RAG_TOP_K,
            county_filter=ctx.county_name,
            zip_filter=ctx.zip_code,
        )

        context_text = self._build_context(chunks)
        user_prompt = (
            f"County: {ctx.county_name or 'California'}\n"
            f"ZIP: {ctx.zip_code or 'N/A'}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {ctx.permit_question}\n\nAnswer:"
        )

        answer = await self.llm.generate(system=PERMIT_SYSTEM, user=user_prompt)
        ctx.permit_answers.append(answer)
        ctx.permit_count += 1
        ctx.state = AgentState.PERMIT_FOLLOWUP

        await self._store_permit_question(
            ctx=ctx,
            chunks=chunks,
            answer=answer,
            state_before=state_before,
        )

        sources = list({c.source_url for c in chunks if c.source_url})
        source_text = ""
        if sources:
            source_text = "\n\n📎 **Sources:** " + " | ".join(f"[{s}]({s})" for s in sources[:3])

        return AgentResponse(
            message=answer + source_text,
            state=ctx.state,
            data={"county": ctx.county_name, "chunks_used": len(chunks)},
            suggestions=["Ask another permit question", "Get renovation suggestions", "Done"],
        )

    async def _store_permit_question(
        self,
        *,
        ctx: ConversationContext,
        chunks: List[PermitChunk],
        answer: str,
        state_before: str,
    ) -> None:
        if not self.question_store.enabled or not ctx.session_id or not ctx.permit_question:
            return

        try:
            await asyncio.to_thread(
                self.question_store.store_permit_question,
                session_id=ctx.session_id,
                question_text=ctx.permit_question,
                county_name=ctx.county_name,
                zip_code=ctx.zip_code,
                city=ctx.city,
                chunks=chunks,
                answer=answer,
                state_before=state_before,
                state_after=ctx.state.value,
                permit_count=ctx.permit_count,
                reno_count=ctx.reno_count,
            )
        except Exception as exc:
            print(f"[DocumentDB] Failed to persist permit question: {exc}")

    async def _handle_permit_followup(self, msg: str, ctx: ConversationContext) -> AgentResponse:
        ml = msg.lower()
        reno_kw = {"renovation", "remodel", "design", "suggest", "idea", "reno", "interior", "decor"}
        done_kw = {"done", "no", "thanks", "bye", "exit", "quit", "finish"}
        more_kw = {"another", "more", "yes", "question", "permit", "how", "what", "when", "where", "fee", "cost"}

        if any(t in ml for t in reno_kw):
            return self._start_reno_flow(ctx)
        if any(t in ml for t in done_kw) and not any(t in ml for t in more_kw):
            ctx.state = AgentState.TRANSITION_TO_RENO
            return AgentResponse(
                message=(
                    "Before you go — would you like **AI renovation suggestions** "
                    "for your California property, complete with design images? 🏠"
                ),
                state=ctx.state,
                suggestions=["Yes, show me renovation ideas!", "No thanks"],
            )

        ctx.permit_question = msg
        ctx.state = AgentState.ANSWERING_PERMIT
        return await self._answer_permit(ctx)

    # ── Reno transition ───────────────────────────────────────────────────────

    async def _handle_reno_transition(self, msg: str, ctx: ConversationContext) -> AgentResponse:
        if any(t in msg.lower() for t in ["yes", "sure", "ok", "yeah", "show", "let", "idea"]):
            return self._start_reno_flow(ctx)
        ctx.state = AgentState.DONE
        return AgentResponse(
            message="Thanks for using the CA Permit & Renovation Agent! Good luck with your project! 🏗️",
            state=ctx.state,
        )

    def _start_reno_flow(self, ctx: ConversationContext) -> AgentResponse:
        ctx.state = AgentState.COLLECT_RENO_AREA
        return AgentResponse(
            message="🏠 **Renovation Advisor** — which area of your home are you renovating?",
            state=ctx.state,
            suggestions=[
                "Kitchen", "Bathroom", "Living Room", "Bedroom",
                "Backyard / Patio", "Facade / Exterior", "ADU / Garage", "Whole House",
            ],
        )

    # ── Reno ──────────────────────────────────────────────────────────────────

    async def _handle_reno_area(self, msg: str, ctx: ConversationContext) -> AgentResponse:
        ctx.reno_area = msg.strip()
        ctx.state = AgentState.COLLECT_RENO_PREFS
        return AgentResponse(
            message=(
                f"Great — **{ctx.reno_area}** renovation.\n\n"
                "Any preferences? Budget range, style (modern / rustic / traditional / minimalist), "
                "specific materials, or timeline."
            ),
            state=ctx.state,
            suggestions=[
                "Budget-friendly, modern",
                "Mid-range, open concept",
                "Premium, smart-home ready",
                "No specific preferences",
            ],
        )

    async def _handle_reno_prefs(self, msg: str, ctx: ConversationContext) -> AgentResponse:
        ctx.reno_prefs = msg.strip()
        ctx.state = AgentState.GENERATING_RENO

        place = ctx.county_name or ctx.city or "California"

        # Step 1 — get structured suggestions (each has its own image_prompt)
        result = await self._get_reno_suggestions(place, ctx.reno_area, ctx.reno_prefs)
        ctx.reno_result = result

        suggestions_data: List[Dict[str, Any]] = result.get("suggestions", [])[: settings.MAX_SUGGESTIONS]

        # Step 2 — generate one collage image, then split it into four images
        collage_image_url: Optional[str] = None
        metadata_url: Optional[str] = None
        collage_s3_key: Optional[str] = None
        metadata_s3_key: Optional[str] = None
        image_source = "disabled"
        if settings.GENERATE_IMAGES and suggestions_data:
            image_urls, collage_image_url, storage_result, image_source = await self._generate_suggestion_images(
                suggestions_data,
                reno_area=ctx.reno_area or "home",
                place=place,
                user_prefs=ctx.reno_prefs or "",
                summary=result.get("summary", ""),
                session_id=ctx.session_id,
            )
            for i, s in enumerate(suggestions_data):
                s["image_url"] = image_urls[i]
            if storage_result:
                metadata_url = storage_result.get("metadata_url")
                collage_s3_key = storage_result.get("collage_s3_key")
                metadata_s3_key = storage_result.get("metadata_s3_key")
        else:
            for s in suggestions_data:
                s["image_url"] = None

        ctx.reno_count += 1
        ctx.state = AgentState.RENO_FOLLOWUP

        summary = result.get("summary", "")

        return AgentResponse(
            message=(
                f"✨ **Renovation ideas for your {ctx.reno_area}** in {place}\n\n"
                f"{summary}\n\n"
                f"Here are **{len(suggestions_data)} design concepts** — browse the gallery below:"
            ),
            state=ctx.state,
            data={
                "suggestions": suggestions_data,
                "collage_image_url": collage_image_url,
                "collage_s3_key": collage_s3_key,
                "metadata_s3_key": metadata_s3_key,
                "metadata_url": metadata_url,
                "image_source": image_source,
                "place": place,
                "house_part": ctx.reno_area,
                "summary": summary,
            },
            suggestions=["Check permits for this renovation", "Get more ideas", "Done"],
        )

    async def _generate_suggestion_images(
        self,
        suggestions: List[Dict[str, Any]],
        reno_area: str,
        place: str,
        user_prefs: str,
        summary: str,
        session_id: Optional[str],
    ) -> tuple[List[Optional[str]], Optional[str], Optional[Dict[str, str]], str]:
        """Reuse a cached collage from S3 or generate a new one and persist it."""
        try:
            cache_key = self.storage.build_cache_key(
                place,
                reno_area,
                user_prefs,
                len(suggestions),
            )

            if self.storage.enabled:
                cached = await asyncio.to_thread(self.storage.get_cached_collage, cache_key)
                if cached:
                    image_urls = await asyncio.to_thread(
                        self._split_collage_into_images_from_bytes,
                        cached["image_bytes"],
                        len(suggestions),
                    )
                    if len(image_urls) < len(suggestions):
                        image_urls.extend([None] * (len(suggestions) - len(image_urls)))
                    return image_urls, cached["collage_url"], {
                        "collage_s3_key": cached["collage_s3_key"],
                        "metadata_s3_key": cached["metadata_s3_key"],
                        "metadata_url": cached["metadata_url"],
                    }, "s3-cache"

            collage_prompt = self._build_collage_prompt(suggestions, reno_area, place)
            collage_data_uri = await self.llm.generate_image(collage_prompt)
            if not collage_data_uri:
                return [None] * len(suggestions), None, None, "none"
            collage_bytes = self._decode_data_uri(collage_data_uri)
            image_urls = await asyncio.to_thread(
                self._split_collage_into_images_from_bytes,
                collage_bytes,
                len(suggestions),
            )
            if len(image_urls) < len(suggestions):
                image_urls.extend([None] * (len(suggestions) - len(image_urls)))
            storage_result = None
            collage_image_url = collage_data_uri
            if self.storage.enabled:
                metadata = self._build_collage_metadata(
                    cache_key=cache_key,
                    place=place,
                    reno_area=reno_area,
                    user_prefs=user_prefs,
                    summary=summary,
                    suggestions=suggestions,
                    session_id=session_id,
                )
                storage_result = await asyncio.to_thread(
                    self.storage.put_collage,
                    cache_key=cache_key,
                    image_bytes=collage_bytes,
                    metadata=metadata,
                )
                collage_image_url = storage_result["collage_url"]
            return image_urls, collage_image_url, storage_result, "generated"
        except Exception as exc:
            print(f"[Agent] Collage image failed: {exc}")
            return [None] * len(suggestions), None, None, "error"

    def _build_collage_prompt(
        self, suggestions: List[Dict[str, Any]], reno_area: str, place: str
    ) -> str:
        quadrant_names = ["top left", "top right", "bottom left", "bottom right"]
        concept_lines = []
        for i, suggestion in enumerate(suggestions[:4]):
            concept_prompt = suggestion.get("image_prompt") or suggestion.get("description") or (
                f"Photorealistic {suggestion.get('style', 'modern')} {suggestion.get('title', 'home renovation')}"
            )
            concept_lines.append(
                f"{quadrant_names[i]} quadrant: {suggestion.get('title', f'Concept {i + 1}')}. "
                f"{concept_prompt}."
            )

        return (
            f"Create exactly one square image containing exactly four square renovation panels for {reno_area} ideas in {place}, California. "
            "Use a strict 2x2 grid: top left, top right, bottom left, bottom right. "
            "Each panel must contain exactly one full renovation scene only, not a collage, not multiple rooms, not extra tiles, and not repeated mini-images. "
            "Make the four panels equal-sized squares with clear gutters between them so they can be cropped cleanly. "
            "Keep one dominant room view centered in each panel with no cut-off furniture at the panel edges. "
            "Do not add any text, labels, letters, numbers, watermarks, frames, captions, or decorative mini-panels. "
            "Keep the viewpoint and lighting polished and photorealistic for an architectural presentation. "
            + " ".join(concept_lines)
            + IMAGE_SUFFIX
        )

    def _split_collage_into_images_from_bytes(
        self, image_bytes: bytes, expected_images: int
    ) -> List[Optional[str]]:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            width, height = img.size
            half_width = width // 2
            half_height = height // 2
            inset_x = max(8, width // 80)
            inset_y = max(8, height // 80)
            boxes = [
                (inset_x, inset_y, half_width - inset_x, half_height - inset_y),
                (half_width + inset_x, inset_y, width - inset_x, half_height - inset_y),
                (inset_x, half_height + inset_y, half_width - inset_x, height - inset_y),
                (half_width + inset_x, half_height + inset_y, width - inset_x, height - inset_y),
            ]
            output: List[Optional[str]] = []
            for box in boxes[:expected_images]:
                tile = img.crop(box)
                side = min(tile.size)
                left = (tile.width - side) // 2
                top = (tile.height - side) // 2
                tile = tile.crop((left, top, left + side, top + side))
                buffer = io.BytesIO()
                tile.save(buffer, format="PNG")
                output.append(
                    "data:image/png;base64,"
                    + base64.b64encode(buffer.getvalue()).decode("ascii")
                )
            return output

    def _decode_data_uri(self, data_uri: str) -> bytes:
        if not data_uri.startswith("data:image"):
            raise ValueError("Expected a data URI from image generation.")
        _, b64_data = data_uri.split(",", 1)
        return base64.b64decode(b64_data)

    def _build_collage_metadata(
        self,
        *,
        cache_key: str,
        place: str,
        reno_area: str,
        user_prefs: str,
        summary: str,
        suggestions: List[Dict[str, Any]],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "cache_key": cache_key,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "place": place,
            "house_part": reno_area,
            "user_prefs": user_prefs,
            "summary": summary,
            "styles": [s.get("style") for s in suggestions if s.get("style")],
            "budget_tiers": [s.get("budget_tier") for s in suggestions if s.get("budget_tier")],
            "image_model": settings.BEDROCK_IMAGE_MODEL_ID
            if settings.is_production and not settings.OPENAI_API_KEY
            else settings.OPENAI_IMAGE_MODEL,
            "suggestions": [
                {
                    "tile_index": idx,
                    "title": s.get("title"),
                    "description": s.get("description"),
                    "style": s.get("style"),
                    "budget_tier": s.get("budget_tier"),
                    "key_materials": s.get("key_materials", []),
                    "estimated_duration": s.get("estimated_duration"),
                    "estimated_cost": s.get("estimated_cost"),
                    "pros": s.get("pros", []),
                    "local_tip": s.get("local_tip"),
                    "image_prompt": s.get("image_prompt"),
                }
                for idx, s in enumerate(suggestions)
            ],
        }

    async def _handle_reno_followup(self, msg: str, ctx: ConversationContext) -> AgentResponse:
        ml = msg.lower()
        permit_kw = {"permit", "need", "require", "legal", "code", "approval"}
        more_kw   = {"more", "another", "different", "other", "idea"}
        done_kw   = {"done", "thanks", "bye", "no", "finish"}

        if any(t in ml for t in permit_kw):
            ctx.state = AgentState.COLLECT_PERMIT_QUESTION
            return AgentResponse(
                message=f"What permit question do you have for your **{ctx.reno_area}** renovation in {ctx.county_name}?",
                state=ctx.state,
                suggestions=[
                    f"{ctx.reno_area} remodel permit requirements",
                    f"Permit fees for {ctx.reno_area}",
                    "Do I need a permit for this?",
                ],
            )
        if any(t in ml for t in more_kw):
            ctx.state = AgentState.COLLECT_RENO_AREA
            return AgentResponse(
                message="Which area would you like ideas for this time?",
                state=ctx.state,
                suggestions=["Kitchen", "Bathroom", "Living Room", "Backyard", "Bedroom"],
            )
        if any(t in ml for t in done_kw):
            ctx.state = AgentState.DONE
            return AgentResponse(
                message="Thanks! Good luck with your California renovation! 🏡✨",
                state=ctx.state,
            )

        ctx.state = AgentState.RENO_FOLLOWUP
        reply = await self.llm.generate(
            system="You are a renovation advisor for California homes. Answer briefly and helpfully.",
            user=msg,
        )
        return AgentResponse(
            message=reply,
            state=ctx.state,
            suggestions=["Get more ideas", "Check permits", "Done"],
        )

    async def _fallback(self, msg: str, ctx: ConversationContext) -> AgentResponse:
        ctx.state = AgentState.COLLECT_LOCATION
        return AgentResponse(
            message="Let me start over — what's your California ZIP code or county?",
            state=ctx.state,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_context(self, chunks: List[PermitChunk]) -> str:
        if not chunks:
            return "No relevant permit data found in local database."
        return "\n\n---\n\n".join(
            f"[Source {i} | {c.county_name} | {c.source_url or 'N/A'}]\n{c.content}"
            for i, c in enumerate(chunks, 1)
        )

    async def _get_reno_suggestions(self, place: str, house_part: str, prefs: str) -> dict:
        user_prompt = RENO_USER_TEMPLATE.format(
            place=place,
            house_part=house_part,
            user_query=prefs or "No specific requirements",
            max_suggestions=settings.MAX_SUGGESTIONS,
        )
        raw = await self.llm.generate(system=RENO_SYSTEM, user=user_prompt)

        # Attempt 1: repair and parse
        try:
            repaired = _repair_truncated_json(raw)
            return json.loads(repaired)
        except Exception as e:
            print(f"[Agent] JSON repair failed: {e}\nRaw response (first 500 chars):\n{raw[:500]}")

        # Attempt 2: ask the LLM to fix its own output
        fix_prompt = (
            "The following JSON is malformed or truncated. "
            "Return ONLY the corrected, complete, valid JSON — nothing else:\n\n" + raw
        )
        try:
            fixed = await self.llm.generate(system="You are a JSON repair tool. Output only valid JSON.", user=fix_prompt)
            fixed = re.sub(r"^```json\s*|^```\s*|```$", "", fixed.strip(), flags=re.MULTILINE).strip()
            return json.loads(fixed)
        except Exception as e2:
            raise ValueError(f"Could not parse renovation suggestions after repair attempt: {e2}")


# ── In-memory session store ───────────────────────────────────────────────────

_sessions: Dict[str, ConversationContext] = {}


def get_session(session_id: str) -> ConversationContext:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationContext(
            session_id=session_id,
            state=AgentState.COLLECT_LOCATION,
        )
    return _sessions[session_id]


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
