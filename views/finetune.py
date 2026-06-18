"""Fine-tuning page — Base vs Fine-tuned headline generation.

Sends the same news description to the un-tuned base Gemini and to the
fine-tuned model, side by side. This is the live proof that fine-tuning
changed the model's behaviour (the assignment's "customization" dimension).
"""

import streamlit as st

import config
from chatbots import LLMResponse, headline_base, headline_tuned


def _render(title: str, result: LLMResponse) -> None:
    st.subheader(title)
    if result.ok:
        st.markdown(f"### {result.text}")
        st.caption(
            f"Model: `{result.model}` · {result.latency_s}s · "
            f"{result.input_tokens} in / {result.output_tokens} out tokens"
        )
    else:
        st.warning(result.error)


def render() -> None:
    st.header("Headline Generator — Base vs Fine-tuned")
    st.markdown(
        "Enter a news description. The same input goes to the **base** Gemini "
        "and to the **fine-tuned** model, so you can see what tuning changed."
    )

    if not config.TUNED_GEMINI_MODEL:
        st.info(
            "Fine-tuned model not configured yet. After the Vertex tuning job "
            "finishes, set `TUNED_GEMINI_MODEL` in your `.env` to the tuned "
            "model's resource name. The base model still works below."
        )

    description = st.text_area(
        "News description:",
        value="A new study finds that people who drink coffee regularly may "
              "have a lower risk of heart disease, according to researchers.",
        height=100,
    )

    if not st.button("Generate Headlines", type="primary"):
        return
    if not description.strip():
        st.warning("Please enter a description first.")
        return

    col_base, col_tuned = st.columns(2)
    with col_base:
        with st.spinner("Base model..."):
            _render(f"Base ({config.HEADLINE_BASE_MODEL})", headline_base(description))
    with col_tuned:
        with st.spinner("Fine-tuned model..."):
            _render("Fine-tuned", headline_tuned(description))
