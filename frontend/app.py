"""
Streamlit MVP Frontend — CA Permit & Renovation Agent.
Theme-adaptive: no hardcoded background/text colours.
Renovation results shown as a colossal full-width sliding gallery —
one large image + detail panel per suggestion, navigated with Prev/Next
and a clickable thumbnail strip.
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
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
    color: #ffffff !important;
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
  }
  .main-header p { color: rgba(255,255,255,0.88); margin: 0.3rem 0 0; }

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

  /* Slider container */
  .slider-wrap {
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 16px;
    overflow: hidden;
    margin: 0.8rem 0 0.4rem;
  }

  /* Detail panel */
  .slide-detail { padding: 1.4rem 1.6rem 1rem; }
  .slide-title  { font-size: 1.5rem; font-weight: 800; margin: 0 0 4px; line-height: 1.2; }
  .slide-sub    { font-size: 0.84rem; opacity: 0.58; margin-bottom: 10px; }
  .slide-desc   { font-size: 0.92rem; opacity: 0.85; line-height: 1.6; margin-bottom: 10px; }
  .slide-meta   { display:flex; gap:1.4rem; flex-wrap:wrap; font-size:0.82rem; opacity:0.68; margin-bottom:8px; }
  .slide-mats   { font-size:0.8rem; opacity:0.62; margin-bottom:8px; }
  .slide-pros   { font-size:0.82rem; opacity:0.75; line-height:1.75; margin-bottom:8px; }
  .slide-tip    { font-size:0.82rem; border-left:3px solid #3dba6e; padding-left:10px; opacity:0.8; }

  /* Budget pill */
  .bpill {
    display:inline-block; font-size:0.7rem; font-weight:700;
    padding:2px 10px; border-radius:20px; border:1.5px solid currentColor;
    margin-bottom:8px; text-transform:uppercase; letter-spacing:0.06em;
  }
  .bp-budget  { color:#1e8c4a; }
  .bp-mid     { color:#9a7200; }
  .bp-premium { color:#b33020; }
  @media (prefers-color-scheme: dark) {
    .bp-budget  { color:#5dd68a; }
    .bp-mid     { color:#e0c040; }
    .bp-premium { color:#f08070; }
  }

  /* Slide counter overlay */
  .slide-counter-bar {
    text-align: right;
    font-size: 0.78rem;
    opacity: 0.55;
    padding: 4px 14px 0;
  }

  /* Thumbnail strip */
  .thumb-label {
    font-size: 0.72rem;
    text-align: center;
    opacity: 0.7;
    margin-top: 3px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
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

if "slide_indices" not in st.session_state:
    st.session_state.slide_indices: dict = {}


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
        st.error(f"Cannot reach backend at **{API_BASE}** — `{exc}`")


def send_message(message: str) -> None:
    if not message.strip():
        return
    st.session_state.messages.append({"role": "user", "content": message})
    st.session_state.quick_replies = []
    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={"session_id": st.session_state.session_id, "message": message},
            timeout=180,   # 5 concurrent DALL-E renders can take ~90 s
        )
        r.raise_for_status()
        data = r.json()
        msg_data = data.get("data")
        clean_msg_data = _strip_image_payloads(msg_data)
        st.session_state.messages.append(
            {"role": "agent", "content": data["message"], "data": clean_msg_data}
        )
        st.session_state.agent_state = data.get("state")
        st.session_state.quick_replies = data.get("suggestions", [])
        if msg_data and "suggestions" in msg_data:
            st.session_state.reno_data = msg_data
    except requests.exceptions.Timeout:
        st.session_state.messages.append({
            "role": "agent",
            "content": "⏱️ Timed out — generating the renovation collage can take up to 60–90 s. Please try again.",
            "data": None,
        })
    except Exception as exc:
        st.session_state.messages.append(
            {"role": "agent", "content": f"❌ Error: {exc}", "data": None}
        )


def reset_session() -> None:
    try:
        requests.post(f"{API_BASE}/session/reset",
                      json={"session_id": st.session_state.session_id}, timeout=10)
    except Exception:
        pass
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.agent_state = None
    st.session_state.reno_data = None
    st.session_state.started = False
    st.session_state.quick_replies = []
    st.session_state.slide_indices = {}


def _pill_cls(budget: str) -> str:
    b = budget.lower()
    if "premium" in b: return "bp-premium"
    if "budget"  in b: return "bp-budget"
    return "bp-mid"


def _strip_image_payloads(data: dict | None) -> dict | None:
    if not data:
        return data
    clean = dict(data)
    clean.pop("collage_image_url", None)
    if "suggestions" in clean:
        clean["suggestions"] = [
            {k: v for k, v in suggestion.items() if k != "image_url"}
            for suggestion in clean.get("suggestions", [])
        ]
    return clean


# ── Colossal slider ────────────────────────────────────────────────────────────
def render_slider(suggestions: list, block_key: str) -> None:
    n = len(suggestions)
    if n == 0:
        return

    sk = f"si_{block_key}"
    if sk not in st.session_state.slide_indices:
        st.session_state.slide_indices[sk] = 0
    idx = max(0, min(st.session_state.slide_indices[sk], n - 1))

    s = suggestions[idx]

    st.markdown('<div class="slider-wrap">', unsafe_allow_html=True)

    # ── Hero image ─────────────────────────────────────────────────────────────
    img_url = s.get("image_url")
    if img_url:
        st.image(img_url, use_container_width=True)
    else:
        st.markdown(
            '<div style="height:420px;display:flex;align-items:center;'
            'justify-content:center;font-size:5rem;opacity:0.2;">🏠</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="slide-counter-bar">Design {idx + 1} of {n}</div>',
        unsafe_allow_html=True,
    )

    # ── Detail panel ───────────────────────────────────────────────────────────
    pill = _pill_cls(s.get("budget_tier", ""))
    pros_html = "<br>".join(f"✓ {p}" for p in s.get("pros", []))
    mats_html = " &nbsp;·&nbsp; ".join(s.get("key_materials", []))
    tip = s.get("local_tip", "")

    st.markdown(f"""
<div class="slide-detail">
  <div class="bpill {pill}">{s.get('budget_tier', '')}</div>
  <div class="slide-title">{s.get('title', '')}</div>
  <div class="slide-sub">{s.get('style', '')} &nbsp;|&nbsp; {s.get('estimated_duration', '')} &nbsp;|&nbsp; {s.get('estimated_cost', '')}</div>
  <div class="slide-desc">{s.get('description', '')}</div>
  <div class="slide-mats">🪵 &nbsp;{mats_html}</div>
  <div class="slide-pros">{pros_html}</div>
  {'<div class="slide-tip">💡 ' + tip + '</div>' if tip else ''}
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close slider-wrap

    # ── Navigation row ─────────────────────────────────────────────────────────
    nav_l, _, nav_r = st.columns([1, 8, 1])
    with nav_l:
        if st.button("◀", key=f"prev_{block_key}", disabled=(idx == 0),
                     use_container_width=True):
            st.session_state.slide_indices[sk] = idx - 1
            st.rerun()
    with nav_r:
        if st.button("▶", key=f"next_{block_key}", disabled=(idx == n - 1),
                     use_container_width=True):
            st.session_state.slide_indices[sk] = idx + 1
            st.rerun()

    # ── Thumbnail strip ────────────────────────────────────────────────────────
    st.markdown("---")
    thumb_cols = st.columns(n)
    for i, sug in enumerate(suggestions):
        with thumb_cols[i]:
            title_short = sug.get("title", f"#{i+1}")
            if len(title_short) > 18:
                title_short = title_short[:17] + "…"
            btn_type = "primary" if i == idx else "secondary"
            if st.button(title_short, key=f"thumb_{block_key}_{i}",
                         use_container_width=True, type=btn_type):
                st.session_state.slide_indices[sk] = i
                st.rerun()


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
        "1. **Enter your CA ZIP or county**\n"
        "2. **Ask permit questions** — offline CA database\n"
        "3. **Request renovation ideas** — 4 designs from one DALL-E collage\n"
        "4. **Swipe / click thumbnails** to browse designs"
    )
    st.divider()
    rdata = st.session_state.reno_data
    if rdata:
        st.markdown(f"### 🎨 {rdata.get('house_part', 'Renovation')}")
        st.caption(f"📍 {rdata.get('place', '')}")
        for i, s in enumerate(rdata.get("suggestions", []), 1):
            with st.expander(f"{i}. {s.get('title', '')}"):
                st.markdown(f"**Style:** {s.get('style', '')}")
                st.markdown(f"**Budget:** {s.get('budget_tier', '')}")
                if s.get("estimated_cost"):
                    st.markdown(f"**Cost:** {s['estimated_cost']}")
                st.markdown(f"**Duration:** {s.get('estimated_duration', '')}")
                st.markdown(f"*{s.get('description', '')}*")
                if s.get("key_materials"):
                    st.markdown("**Materials:** " + ", ".join(s["key_materials"]))
                if s.get("local_tip"):
                    st.info(f"💡 {s['local_tip']}")


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1 style="margin:0;font-size:1.75rem;">🏡 California Permit &amp; Renovation Agent</h1>
  <p>Permit guidance from offline CA data &middot; 4 AI renovation concepts &middot; One DALL-E collage split into 4 images</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.started:
    start_session()

# ── Messages ───────────────────────────────────────────────────────────────────
for msg_idx, msg in enumerate(st.session_state.messages):
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
        mdata = msg.get("data") or {}
        if (
            "suggestions" in mdata
            and mdata["suggestions"]
            and st.session_state.reno_data
            and msg_idx == len(st.session_state.messages) - 1
        ):
            render_slider(st.session_state.reno_data["suggestions"], block_key=str(msg_idx))

# ── Quick replies ──────────────────────────────────────────────────────────────
if st.session_state.quick_replies:
    st.markdown("**Quick replies:**")
    cc = st.columns(min(4, len(st.session_state.quick_replies)))
    for i, reply in enumerate(st.session_state.quick_replies):
        with cc[i % len(cc)]:
            if st.button(reply, key=f"qr_{i}_{reply}", use_container_width=True):
                send_message(reply)
                st.rerun()

# ── Input ──────────────────────────────────────────────────────────────────────
st.divider()
with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    with c1:
        form_input = st.text_input(
            "Message", placeholder="Type your question or CA ZIP code…",
            label_visibility="collapsed",
        )
    with c2:
        submitted = st.form_submit_button("Send →", use_container_width=True, type="primary")

if submitted and form_input:
    send_message(form_input)
    st.rerun()

if st.session_state.reno_data and st.session_state.reno_data.get("collage_image_url"):
    st.divider()
    st.markdown("### Full Collage")
    st.caption("Original 2x2 renovation image used to create the four design tiles.")
    st.image(st.session_state.reno_data["collage_image_url"], use_container_width=True)
