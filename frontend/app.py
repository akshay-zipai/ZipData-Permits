"""
Streamlit MVP Frontend — CA Permit & Renovation Agent.
Theme-adaptive: no hardcoded background/text colours; works on both
Streamlit light and dark themes.
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CA Permit & Renovation Agent",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
# Rules that set background or text colour use either:
#   • a CSS variable from Streamlit's own theme  (var(--text-color) etc.)
#   • opacity-based dimming on the inherited colour
#   • explicit dark-mode overrides via @media (prefers-color-scheme: dark)
# No raw #hex values on content surfaces.
st.markdown("""
<style>
  /* Header — intentionally always dark-bg so white text is fine */
  .main-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
    color: #ffffff !important;
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
  }
  .main-header p { color: rgba(255,255,255,0.88); margin: 0.3rem 0 0; }

  /* Chat bubbles — coloured left border only; bg inherits theme surface */
  .chat-bubble-user {
    border-left: 4px solid #4a9fd4;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.4rem 0;
  }
  .chat-bubble-agent {
    border-left: 4px solid #3dba6e;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.4rem 0;
  }
  .chat-bubble-user strong,
  .chat-bubble-agent strong {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.65;
  }

  /* Renovation suggestion cards — border only, no background */
  .suggestion-card {
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.4rem 0;
    height: 100%;
  }
  .card-title {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 4px;
  }
  /* Budget meta — text colour only, readable on light & dark */
  .card-meta { font-size: 0.78rem; font-weight: 600; margin-bottom: 6px; opacity: 0.9; }
  .meta-budget  { color: #1e8c4a; }
  .meta-mid     { color: #9a7200; }
  .meta-premium { color: #b33020; }
  @media (prefers-color-scheme: dark) {
    .meta-budget  { color: #5dd68a; }
    .meta-mid     { color: #e0c040; }
    .meta-premium { color: #f08070; }
  }
  .card-desc     { font-size: 0.83rem; opacity: 0.82; }
  .card-duration { font-size: 0.78rem; opacity: 0.58; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("session_id", str(uuid.uuid4())),
    ("messages", []),
    ("agent_state", None),
    ("reno_data", None),
    ("started", False),
    ("quick_replies", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── API helpers ────────────────────────────────────────────────────────────────
def start_session() -> None:
    try:
        r = requests.post(
            f"{API_BASE}/session/start",
            json={"session_id": st.session_state.session_id},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        st.session_state.messages.append(
            {"role": "agent", "content": data["message"], "data": data.get("data")}
        )
        st.session_state.agent_state = data.get("state")
        st.session_state.quick_replies = data.get("suggestions", [])
        st.session_state.started = True
    except Exception as exc:
        st.error(
            f"Cannot reach the agent backend at **{API_BASE}**.\n\n"
            f"Error: `{exc}`\n\nStart the API server first, then refresh this page."
        )


def send_message(message: str) -> None:
    if not message.strip():
        return
    st.session_state.messages.append({"role": "user", "content": message})
    st.session_state.quick_replies = []

    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={"session_id": st.session_state.session_id, "message": message},
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        msg_data = data.get("data")
        st.session_state.messages.append(
            {"role": "agent", "content": data["message"], "data": msg_data}
        )
        st.session_state.agent_state = data.get("state")
        st.session_state.quick_replies = data.get("suggestions", [])
        if msg_data and "suggestions" in msg_data:
            st.session_state.reno_data = msg_data
    except requests.exceptions.Timeout:
        st.session_state.messages.append({
            "role": "agent",
            "content": "⏱️ Request timed out (image generation can take up to 30 s). Please try again.",
            "data": None,
        })
    except Exception as exc:
        st.session_state.messages.append(
            {"role": "agent", "content": f"❌ Error: {exc}", "data": None}
        )


def reset_session() -> None:
    try:
        requests.post(
            f"{API_BASE}/session/reset",
            json={"session_id": st.session_state.session_id},
            timeout=10,
        )
    except Exception:
        pass
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.agent_state = None
    st.session_state.reno_data = None
    st.session_state.started = False
    st.session_state.quick_replies = []


def _budget_class(budget_tier: str) -> str:
    t = budget_tier.lower()
    if "premium" in t:
        return "meta-premium"
    if "budget" in t:
        return "meta-budget"
    return "meta-mid"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏡 Agent Controls")
    st.caption(f"Session `{st.session_state.session_id[:8]}…`")

    if st.session_state.agent_state:
        st.caption(f"State: `{st.session_state.agent_state}`")

    if st.button("🔄 New Session", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    st.markdown("### 📋 How it works")
    st.markdown(
        "1. **Enter your CA ZIP or county** to start\n"
        "2. **Ask permit questions** — answered from 20 k+ offline CA records\n"
        "3. **Request renovation ideas** — AI suggestions + DALL-E visualization\n"
        "4. **Switch topics** at any point"
    )
    st.divider()

    # Renovation results panel
    rdata = st.session_state.reno_data
    if rdata:
        st.markdown(f"### 🎨 {rdata.get('house_part', 'Renovation')}")
        st.caption(f"📍 {rdata.get('place', '')}")
        if rdata.get("image_url"):
            img_url = rdata["image_url"]
            # Handle both URL strings and base64 data-URIs (Bedrock Titan)
            st.image(img_url, caption="AI Design Visualization", use_container_width=True)
        for i, s in enumerate(rdata.get("suggestions", []), 1):
            with st.expander(f"{i}. {s.get('title', '')}"):
                st.markdown(f"**Style:** {s.get('style', '')}")
                st.markdown(f"**Budget:** {s.get('budget_tier', '')}")
                if s.get("estimated_cost"):
                    st.markdown(f"**Est. Cost:** {s['estimated_cost']}")
                st.markdown(f"**Duration:** {s.get('estimated_duration', '')}")
                st.markdown(f"*{s.get('description', '')}*")
                if s.get("key_materials"):
                    st.markdown("**Materials:** " + ", ".join(s["key_materials"]))
                if s.get("pros"):
                    for p in s["pros"]:
                        st.markdown(f"  • {p}")
                if s.get("local_tip"):
                    st.info(f"💡 **CA Tip:** {s['local_tip']}")


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1 style="margin:0;font-size:1.75rem;">🏡 California Permit &amp; Renovation Agent</h1>
  <p>Permit guidance from offline CA data &middot; AI renovation ideas &middot; Design visualizations</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.started:
    start_session()

# ── Chat messages ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-bubble-user"><strong>You</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-bubble-agent"><strong>🤖 Agent</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        # Inline renovation cards
        mdata = msg.get("data") or {}
        if "suggestions" in mdata:
            slist = mdata["suggestions"]
            cols = st.columns(min(3, max(1, len(slist))))
            for i, s in enumerate(slist):
                bc = _budget_class(s.get("budget_tier", "mid"))
                with cols[i % len(cols)]:
                    st.markdown(
                        f"""<div class="suggestion-card">
  <div class="card-title">{s.get('title', '')}</div>
  <div class="card-meta {bc}">{s.get('budget_tier', '')} &middot; {s.get('style', '')}</div>
  <div class="card-desc">{s.get('description', '')}</div>
  <div class="card-duration">⏱ {s.get('estimated_duration', '')}</div>
</div>""",
                        unsafe_allow_html=True,
                    )
            if mdata.get("image_url"):
                st.image(
                    mdata["image_url"],
                    caption="🎨 AI Design Visualization",
                    use_container_width=True,
                )

# ── Quick-reply chips ──────────────────────────────────────────────────────────
if st.session_state.quick_replies:
    st.markdown("**Quick replies:**")
    chip_cols = st.columns(min(4, len(st.session_state.quick_replies)))
    for i, reply in enumerate(st.session_state.quick_replies):
        with chip_cols[i % len(chip_cols)]:
            if st.button(reply, key=f"qr_{i}_{reply}", use_container_width=True):
                send_message(reply)
                st.rerun()

# ── Input ──────────────────────────────────────────────────────────────────────
st.divider()
with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    with c1:
        form_input = st.text_input(
            "Message",
            placeholder="Type your question or CA ZIP code…",
            label_visibility="collapsed",
        )
    with c2:
        submitted = st.form_submit_button("Send →", use_container_width=True, type="primary")

if submitted and form_input:
    send_message(form_input)
    st.rerun()
