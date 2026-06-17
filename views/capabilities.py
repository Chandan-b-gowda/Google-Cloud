"""Capabilities page — Gemini Image Generation and Speech-to-Text.

Two additional Gemini capabilities beyond the plain text generation
covered elsewhere in the app. Each section is self-contained: a small
UI, a call into chatbots.gemini_bot, and a rendered result with
timing/usage metrics.
"""

import streamlit as st

import config
from chatbots import generate_image, transcribe_audio


def _render_image_section() -> None:
    st.subheader("🖼️ Image Generation")
    st.caption(f"Model: `{config.GEMINI_IMAGE_MODEL}`")

    prompt = st.text_area(
        "Describe the image you want:",
        value="A watercolor painting of a lighthouse at sunset.",
        height=80,
        key="image_prompt",
    )

    if st.button("Generate Image", type="primary"):
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


def _render_audio_section() -> None:
    st.subheader("🎙️ Speech-to-Text")
    st.caption(f"Model: `{config.GEMINI_AUDIO_MODEL}`")
    st.markdown("Record a short clip with your microphone, then transcribe it.")

    audio = st.audio_input("Record audio")

    if audio is not None:
        st.audio(audio)
        if st.button("Transcribe", type="primary"):
            with st.spinner("Gemini is listening..."):
                audio_bytes = audio.getvalue()
                result = transcribe_audio(audio_bytes, mime_type="audio/wav")

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
    st.header("✨ Gemini Capabilities")
    st.markdown(
        "Two Gemini capabilities beyond text generation: **image "
        "generation** and **speech-to-text**, both wired into the UI."
    )

    _render_image_section()
    st.markdown("---")
    _render_audio_section()
