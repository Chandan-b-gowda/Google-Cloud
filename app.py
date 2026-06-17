"""DoubleChat — Multi-LLM Comparison Lab.

Streamlit entry point. Run with:  streamlit run app.py

SRH Advanced Computer Science — Final Exam Project (Topic K:
comparison of Google Gemini, Anthropic Claude, and OpenAI ChatGPT).

This file owns page configuration and navigation. Each page's content
lives in its own module under views/ (compare, chat, history).
"""

import streamlit as st

import config
from views import capabilities, chat, compare, history

# st.set_page_config must be the first Streamlit call, and run only once.
st.set_page_config(
    page_title="DoubleChat — LLM Comparison Lab",
    page_icon="🤖",
    layout="wide",
)

PAGES = {
    "Compare": compare.render,
    "Capabilities": capabilities.render,
    "Chat": chat.render,
    "History": history.render,
}

# --- Navigation state -------------------------------------------------------
# Default landing page.
st.session_state.setdefault("page", "Compare")

# Honor a programmatic navigation request from another page (e.g. History's
# "Load into Chat"). We apply it *before* the nav widget is created, because
# Streamlit forbids mutating a widget-bound key after the widget exists.
if "_nav_to" in st.session_state:
    st.session_state["page"] = st.session_state.pop("_nav_to")

# --- Sidebar: navigation + live configuration status ------------------------
with st.sidebar:
    st.title("🤖 DoubleChat")
    st.caption("LLM Comparison Lab")

    st.radio("Navigate", options=list(PAGES.keys()), key="page")

    st.markdown("---")
    st.subheader("⚙️ Configuration")
    problems = config.validate()
    if problems:
        for problem in problems:
            st.error(problem)
    else:
        st.success("Configuration OK")

    st.markdown("**Gemini** (Vertex AI)")
    st.code(config.GEMINI_MODEL, language=None)
    st.caption(f"project `{config.GCP_PROJECT}`")

    st.markdown("**Claude** (Anthropic API)")
    st.code(config.CLAUDE_MODEL, language=None)

# --- Dispatch to the selected page ------------------------------------------
if problems:
    st.title("🤖 DoubleChat — LLM Comparison Lab")
    st.error(
        "Configuration is incomplete — fix the issues listed in the sidebar, "
        "then reload."
    )
else:
    PAGES[st.session_state["page"]]()
