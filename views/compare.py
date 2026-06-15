"""Compare page — send one prompt to both models, show answers side by side."""

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
    st.header("🚀 Side-by-Side Comparison")
    st.markdown(
        "Send the same prompt to both models and compare their answers, "
        "latency, and token usage."
    )

    prompt = st.text_area(
        "Your prompt:",
        value="Explain prompt caching in large language models in 2-3 sentences.",
        height=100,
    )

    if st.button("Ask Both", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
            return

        col_gemini, col_claude = st.columns(2)
        with col_gemini:
            with st.spinner("Gemini thinking..."):
                gemini_result = ask_gemini(prompt)
            _render_response("🟦 Gemini", gemini_result)
        with col_claude:
            with st.spinner("Claude thinking..."):
                claude_result = ask_claude(prompt)
            _render_response("🟧 Claude", claude_result)
