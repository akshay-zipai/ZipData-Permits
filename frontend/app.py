import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="House Renovation Advisor",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 House Renovation Advisor")
st.caption("AI-powered renovation suggestions tailored to your location and space.")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Details")

    place = st.text_input("📍 Location", placeholder="e.g. Jaipur, India")
    house_part = st.selectbox(
        "🏗️ Part of House",
        ["Living Room", "Kitchen", "Bedroom", "Bathroom", "Facade / Exterior",
         "Balcony", "Dining Room", "Home Office", "Garden", "Other"],
    )
    custom_part = ""
    if house_part == "Other":
        custom_part = st.text_input("Specify part", placeholder="e.g. Staircase")

    query = st.text_area(
        "💬 Additional preferences",
        placeholder="e.g. Budget-friendly, prefer natural materials, Rajasthani style...",
        height=100,
    )
    generate_image = st.toggle("🎨 Generate visualization image", value=True)
    submit = st.button("Get Suggestions", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
if submit:
    part = custom_part if house_part == "Other" else house_part

    if not place or not part:
        st.warning("Please fill in your location and the part of the house.")
        st.stop()

    with st.spinner("Generating renovation suggestions..."):
        try:
            resp = requests.post(
                f"{API_URL}/renovation/suggest",
                json={
                    "place": place,
                    "house_part": part,
                    "query": query,
                    "generate_image": generate_image,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure the backend is running.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.text}")
            st.stop()

    # Summary
    st.subheader(f"Renovation Ideas for your {data['house_part']} in {data['place']}")
    st.info(data["summary"])

    # Image
    if data.get("image_url"):
        st.image(data["image_url"], caption="AI-generated visualization", use_container_width=True)

    st.divider()

    # Suggestions
    st.subheader(f"💡 {len(data['suggestions'])} Suggestions")

    for i, s in enumerate(data["suggestions"], 1):
        with st.expander(f"{i}. {s['title']}  •  {s['budget_tier']}", expanded=i == 1):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(s["description"])
                st.markdown(f"**🏷️ Style:** {s['style']}")
                st.markdown(f"**⏱️ Duration:** {s['estimated_duration']}")
                if s.get("pros"):
                    st.markdown("**✅ Pros:**")
                    for p in s["pros"]:
                        st.markdown(f"- {p}")

            with col2:
                st.markdown("**🧱 Key Materials:**")
                for m in s.get("key_materials", []):
                    st.markdown(f"- {m}")
                if s.get("local_tip"):
                    st.markdown("**📌 Local Tip:**")
                    st.caption(s["local_tip"])

else:
    st.markdown(
        """
        ### How it works
        1. Enter your **location** — suggestions are tailored to local climate, culture & materials
        2. Pick the **part of the house** you want to renovate
        3. Add any **style or budget preferences**
        4. Get **AI-powered suggestions** + an optional DALL-E visualization
        """
    )
