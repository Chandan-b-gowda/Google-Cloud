"""Capabilities page — Gemini's multimodal features, shown as side-by-side
tabs: image generation and speech-to-text.

These are Gemini-only: Claude has no image output and accepts no audio
input, so this page focuses on what Gemini can do beyond plain text.
"""

import streamlit as st

import config
from chatbots import generate_image, transcribe_audio


def _image_tab() -> None:
    st.caption(f"Model: `{config.GEMINI_IMAGE_MODEL}`")

    prompt = st.text_area(
        "Describe the image you want:",
        value="A watercolor painting of a lighthouse at sunset.",
        height=90,
        key="cap_image_prompt",
    )

    if st.button("Generate Image", type="primary", key="cap_image_btn"):
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
            return
        with st.spinner("Gemini is drawing..."):
            result = generate_image(prompt)

        if result.ok:
            st.image(result.image_bytes, caption=prompt, width="stretch")
            if result.text:
                st.markdown(result.text)
            st.caption(
                f"`{result.model}` · {result.latency_s}s · "
                f"{result.input_tokens} in / {result.output_tokens} out tokens"
            )
        else:
            st.error(result.error)


def _voice_tab() -> None:
    st.caption(f"Model: `{config.GEMINI_AUDIO_MODEL}`")
    st.markdown("Record a short clip with your microphone, then transcribe it.")

    audio = st.audio_input("Record audio", key="cap_voice")
    if audio is not None:
        st.audio(audio)
        if st.button("Transcribe", type="primary", key="cap_voice_btn"):
            with st.spinner("Gemini is listening..."):
                result = transcribe_audio(audio.getvalue(), mime_type="audio/wav")

            if result.ok:
                st.markdown(f"**Transcript:** {result.text}")
                st.caption(
                    f"`{result.model}` · {result.latency_s}s · "
                    f"{result.input_tokens} in / {result.output_tokens} out tokens"
                )
            else:
                st.error(result.error)


def render() -> None:
    st.header("Gemini Capabilities")
    st.caption(
        "Multimodal features unique to Gemini — Claude has no image output "
        "or audio input."
    )

    tab_image, tab_voice = st.tabs(["Image Generation", "Speech-to-Text"])
    with tab_image:
        _image_tab()
    with tab_voice:
        _voice_tab()
