"""
Conversational Agent — State machine guiding users through:
  1. Permit Q&A  (location → county/zip → question → RAG answer)
  2. Renovation  (area → preferences → AI suggestions + 5 images, one per suggestion)

LLM calls go through agent.llm.get_llm_client() which transparently
dispatches to OpenAI (local) or AWS Bedrock (production).
"""
from __future__ import annotations

import asyncio
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from agent.config import get_settings
from agent.llm import get_llm_client
from agent.permit_kb import get_permit_kb, PermitChunk

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
Respond ONLY with valid JSON — no markdown fences, no preamble, no trailing text."""

RENO_USER_TEMPLATE = """
I need {max_suggestions} renovation suggestions for:
- Location: {place} (California)
- Area of House: {house_part}
- Additional Requirements: {user_query}

Respond ONLY with valid JSON matching this structure exactly:
{{
  "place": "{place}",
  "house_part": "{house_part}",
  "summary": "1-2 sentence overview of the renovation approach for this region and space",
  "suggestions": [
    {{
      "title": "Short catchy title",
      "description": "2-3 sentence description of the renovation idea",
      "style": "Design style (e.g. Modern, Traditional, Rustic, Minimalist)",
      "budget_tier": "Budget / Mid-range / Premium",
      "key_materials": ["material1", "material2", "material3"],
      "estimated_duration": "e.g. 2-4 weeks",
      "estimated_cost": "e.g. $5,000-$15,000",
      "pros": ["pro1", "pro2", "pro3"],
      "local_tip": "Specific tip relevant to California climate or building codes",
      "image_prompt": "Detailed DALL-E prompt: photorealistic render of a {house_part} in {place} California showing EXACTLY this style and these materials, professional interior design photography, natural lighting"
    }}
  ]
}}
"""

IMAGE_SUFFIX = (
    ", photorealistic architectural render, professional interior design photography, "
    "high resolution, natural lighting, 4K quality, California home, ultra detailed"
)


# ── Agent ─────────────────────────────────────────────────────────────────────

class PermitRenoAgent:

    def __init__(self):
        self.llm = get_llm_client()
        self.kb  = get_permit_kb()

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

        suggestions_data: List[Dict[str, Any]] = result.get("suggestions", [])

        # Step 2 — generate one image per suggestion concurrently
        if settings.GENERATE_IMAGES and suggestions_data:
            image_urls = await self._generate_suggestion_images(suggestions_data)
            for i, s in enumerate(suggestions_data):
                s["image_url"] = image_urls[i]
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
                "place": place,
                "house_part": ctx.reno_area,
                "summary": summary,
            },
            suggestions=["Check permits for this renovation", "Get more ideas", "Done"],
        )

    async def _generate_suggestion_images(
        self, suggestions: List[Dict[str, Any]]
    ) -> List[Optional[str]]:
        """Generate one image per suggestion concurrently."""

        async def _gen_one(s: Dict[str, Any]) -> Optional[str]:
            prompt = s.get("image_prompt") or (
                f"Photorealistic interior design render of a {s.get('style', 'modern')} "
                f"{s.get('title', 'home renovation')}, California home, "
                f"featuring {', '.join(s.get('key_materials', []))}"
            )
            try:
                return await self.llm.generate_image(prompt + IMAGE_SUFFIX)
            except Exception as exc:
                print(f"[Agent] Image failed for '{s.get('title')}': {exc}")
                return None

        return list(await asyncio.gather(*[_gen_one(s) for s in suggestions]))

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
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        return json.loads(raw)


# ── In-memory session store ───────────────────────────────────────────────────

_sessions: Dict[str, ConversationContext] = {}


def get_session(session_id: str) -> ConversationContext:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationContext(state=AgentState.COLLECT_LOCATION)
    return _sessions[session_id]


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)