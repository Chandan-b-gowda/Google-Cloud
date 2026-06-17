"""Compare page — one prompt sent to both models, answers side by side.

Text only. Gemini's image/audio features live on the Capabilities page.
"""

import streamlit as st

from chatbots import LLMResponse, ask_claude, ask_gemini


def _render_response(title: str, result: LLMResponse) -> None:
    """Render one model's answer (or error) plus its metrics."""
    st.subheader(title)
    if result.ok:
        st.markdown(result.text)
        st.caption(
            f"Provider: {result.provider} · Model: `{result.model}` · "
            f"Latency: **{result.latency_s}s** · "
            f"Tokens: {result.input_tokens} in / {result.output_tokens} out"
        )
    else:
        st.error(result.error)


def render() -> None:
    st.markdown(
        "<h1 style='text-align:center;margin-bottom:0;'>Compare</h1>"
        "<p style='text-align:center;color:#9aa0a6;margin-top:4px;'>"
        "Ask the same question and see how Gemini and Claude answer.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    _, center, _ = st.columns([1, 2, 1])
    with center:
        prompt = st.text_area(
            "prompt", key="cmp_prompt", label_visibility="collapsed",
            placeholder="Ask anything — both models will answer…", height=100,
        )
        go = st.button("Ask Both", type="primary", width="stretch")

    if not go:
        return
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
        return

    col_gemini, col_claude = st.columns(2)
    with col_gemini:
        with st.spinner("Gemini thinking..."):
            result = ask_gemini(prompt)
        _render_response("Gemini", result)
    with col_claude:
        with st.spinner("Claude thinking..."):
            result = ask_claude(prompt)
        _render_response("Claude", result)
