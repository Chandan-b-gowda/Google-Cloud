"""DoubleChat — Multi-LLM Comparison Lab.

Streamlit entry point. Run with:  streamlit run app.py

SRH Advanced Computer Science — Final Exam Project (Topic K:
comparison of Google Gemini, Anthropic Claude, and OpenAI ChatGPT).

This file owns page configuration and navigation. Each page's content
lives in its own module under views/.
"""

import streamlit as st

import config
from views import (
    benchmark,
    capabilities,
    chat,
    code_gen,
    compare,
    finetune,
    history,
)

# st.set_page_config must be the first Streamlit call, and run only once.
st.set_page_config(
    page_title="DoubleChat — LLM Comparison Lab",
    layout="wide",
)

PAGES = {
    "Compare": compare.render,
    "Capabilities": capabilities.render,
    "Code Generation": code_gen.render,
    "Chat": chat.render,
    "Benchmark": benchmark.render,
    "Fine-tuned": finetune.render,
    "History": history.render,
}

# --- Navigation state -------------------------------------------------------
st.session_state.setdefault("page", "Compare")

# Honor a programmatic navigation request from another page (e.g. History's
# "Load into Chat"). Applied before the nav widget is created, because
# Streamlit forbids mutating a widget-bound key after the widget exists.
if "_nav_to" in st.session_state:
    st.session_state["page"] = st.session_state.pop("_nav_to")

problems = config.validate()

# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.title("DoubleChat")
    st.caption("Compare Gemini and Claude, side by side.")

    st.radio("Menu", options=list(PAGES.keys()), key="page",
             label_visibility="collapsed")

    with st.expander("Status & models", expanded=False):
        if problems:
            for problem in problems:
                st.error(problem)
        else:
            st.success("Connected")
        st.write(f"Gemini · `{config.GEMINI_MODEL}`")
        st.write(f"Claude · `{config.CLAUDE_MODEL}`")
        st.write(f"Project · `{config.GCP_PROJECT}`")

# --- Render the selected page -----------------------------------------------
if problems:
    st.title("DoubleChat")
    st.error("Setup incomplete — check the sidebar status panel, then reload.")
else:
    PAGES[st.session_state["page"]]()
