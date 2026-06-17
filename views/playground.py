"""Playground page — one place to try everything, organized into tabs like a
typical AI app:

    Compare  — same prompt to both models, answers side by side
    Image    — Gemini text-to-image
    Voice    — Gemini speech-to-text

Combines what used to be the separate Compare and Capabilities pages.
"""

import streamlit as st

import config
from chatbots import (
    LLMResponse,
    ask_claude,
    ask_gemini,
    generate_image,
    transcribe_audio,
)


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


def _compare_tab() -> None:
    st.markdown(
        "Send the same prompt to both models and compare their answers, "
        "latency, and token usage."
    )
    prompt = st.text_area(
        "Your prompt:",
        value="Explain prompt caching in large language models in 2-3 sentences.",
        height=100,
        key="compare_prompt",
    )
    if st.button("Ask Both", type="primary", key="compare_btn"):
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


def _image_tab() -> None:
    st.markdown("Generate an image from a text description using Gemini.")
    st.caption(f"Model: `{config.GEMINI_IMAGE_MODEL}`")

    prompt = st.text_area(
        "Describe the image you want:",
        value="A watercolor painting of a lighthouse at sunset.",
        height=80,
        key="image_prompt",
    )
    if st.button("Generate Image", type="primary", key="image_btn"):
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
            return
        with st.spinner("Gemini is drawing..."):
            result = generate_image(prompt)

        if result.ok:
            st.image(result.image_bytes, caption=prompt)
            if result.text:
                st.markdown(result.text)
            st.caption(
                f"Provider: {result.provider} · Model: `{result.model}` · "
                f"Latency: **{result.latency_s}s** · "
                f"Tokens: {result.input_tokens} in / {result.output_tokens} out"
            )
        else:
            st.error(result.error)


def _voice_tab() -> None:
    st.markdown("Record a short clip with your microphone, then transcribe it.")
    st.caption(f"Model: `{config.GEMINI_AUDIO_MODEL}`")

    audio = st.audio_input("Record audio", key="voice_input")
    if audio is not None:
        st.audio(audio)
        if st.button("Transcribe", type="primary", key="voice_btn"):
            with st.spinner("Gemini is listening..."):
                result = transcribe_audio(audio.getvalue(), mime_type="audio/wav")

            if result.ok:
                st.markdown(f"**Transcript:** {result.text}")
                st.caption(
                    f"Provider: {result.provider} · Model: `{result.model}` · "
                    f"Latency: **{result.latency_s}s** · "
                    f"Tokens: {result.input_tokens} in / {result.output_tokens} out"
                )
            else:
                st.error(result.error)


def render() -> None:
    st.header("🧪 Playground")

    tab_compare, tab_image, tab_voice = st.tabs(
        ["💬 Compare", "🖼️ Image", "🎙️ Voice"]
    )
    with tab_compare:
        _compare_tab()
    with tab_image:
        _image_tab()
    with tab_voice:
        _voice_tab()
