"""
SHL Assessment Recommender - Streamlit Demo Client
Enhanced local testing frontend with FastAPI backend integration.
Optimized for deployment on Streamlit Cloud.
"""

from __future__ import annotations

import os
from typing import Dict, List, Any

import requests
import streamlit as st
from dotenv import load_dotenv

# Optional Groq integration for response polishing
try:
    from groq import Groq
except ImportError:
    Groq = None


# ========================= ENV SETUP =========================
load_dotenv()

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
# Production Backend (Render) - Change only if needed
BACKEND_URL = os.getenv(
    "BACKEND_URL", 
    "https://shl-recommender-api-3rd3.onrender.com"   # ← Your actual Render URL
)
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

groq_client = None
if GROQ_API_KEY and Groq:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        groq_client = None


# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 SHL Conversational Assessment Recommender")
st.markdown("**Talk naturally** with the agent to get the best SHL Individual Test Solutions.")


# ========================= HELPERS =========================
def enhance_with_groq(reply: str) -> str:
    """Optional response polishing via Groq."""
    if not groq_client:
        return reply

    try:
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an SHL assessment recommendation assistant. "
                        "Rewrite responses clearly and professionally without "
                        "changing factual recommendations or URLs."
                    ),
                },
                {"role": "user", "content": reply},
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return reply


def render_recommendations(recommendations: List[Dict[str, Any]]) -> None:
    """Render recommendation cards."""
    if not recommendations:
        return

    st.markdown("### 📋 Recommendations")
    for rec in recommendations:
        name = rec.get("name", "Unknown Assessment")
        url = rec.get("url", "#")
        test_type = rec.get("test_type", "N/A")

        st.markdown(
            f"**{name}**  \n"
            f"🔗 [{url}]({url})  \n"
            f"Type: `{test_type}`"
        )


# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("⚙️ Configuration")

    backend_url = st.text_input(
        "Backend URL", 
        value=BACKEND_URL,
        help="Your Render FastAPI URL"
    )

    st.divider()

    # API Status
    st.subheader("🔑 Integrations")
    if GROQ_API_KEY:
        st.success("✅ Groq API Key Loaded")
    else:
        st.info("Groq API Key Not Found (.env optional)")

    st.divider()

    # Health Check
    if st.button("🔍 Check Backend Health", use_container_width=True):
        try:
            response = requests.get(f"{backend_url}/health", timeout=10)
            if response.status_code == 200:
                st.success("✅ Backend is healthy")
                st.json(response.json())
            else:
                st.error(f"❌ Backend error: {response.status_code}")
        except Exception as exc:
            st.error(f"❌ Connection failed: {exc}")

    st.divider()

    # Reset Conversation
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    debug_mode = st.checkbox("🐞 Debug Mode", value=False)


# ========================= SESSION STATE =========================
if "messages" not in st.session_state:
    st.session_state.messages = []


# ========================= CHAT HISTORY =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("recommendations"):
            render_recommendations(msg["recommendations"])


# ========================= USER INPUT =========================
prompt = st.chat_input("Describe the role you are hiring for...")

if prompt:
    # Add user message
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                payload = {
                    "messages": [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                }

                if debug_mode:
                    st.caption("Payload sent to backend:")
                    st.json(payload)

                response = requests.post(
                    f"{backend_url}/chat",
                    json=payload,
                    timeout=45,
                )

                if response.status_code != 200:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")
                else:
                    data = response.json()

                    reply = data.get("reply", "No reply received.")
                    recommendations = data.get("recommendations", [])
                    end_of_conversation = data.get("end_of_conversation", False)

                    final_reply = enhance_with_groq(reply)

                    st.markdown(final_reply)

                    if recommendations:
                        render_recommendations(recommendations)

                    # Save to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_reply,
                        "recommendations": recommendations,
                    })

                    if end_of_conversation:
                        st.success("🎯 Conversation completed successfully!")

                    if debug_mode:
                        st.caption("Raw backend response:")
                        st.json(data)

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Please check Backend URL.")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. Backend may be slow.")
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")


# ========================= FOOTER =========================
st.divider()
st.caption("SHL AI Intern Assignment | Streamlit Frontend for Testing")
