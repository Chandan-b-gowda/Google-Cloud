"""Playground page — a single centered prompt with mode toggles, in the style
of modern AI apps. One input, three modes:

    Compare  — same prompt to both models, answers side by side
    Image    — Gemini text-to-image
    Voice    — Gemini speech-to-text

Replaces the old separate Compare and Capabilities pages.
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

COMPARE = "💬 Compare"
IMAGE = "🖼️ Image"
VOICE = "🎙️ Voice"


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
    # --- Centered title ------------------------------------------------------
    st.markdown(
        "<h1 style='text-align:center;margin-bottom:0;'>DoubleChat</h1>"
        "<p style='text-align:center;color:#9aa0a6;margin-top:4px;'>"
        "Compare two models, generate an image, or use your voice.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    # --- Centered input area -------------------------------------------------
    prompt: str | None = None
    audio = None
    go = False
    mode = COMPARE

    _, center, _ = st.columns([1, 2, 1])
    with center:
        mode = st.segmented_control(
            "Mode",
            options=[COMPARE, IMAGE, VOICE],
            default=COMPARE,
            label_visibility="collapsed",
        ) or COMPARE

        if mode == IMAGE:
            prompt = st.text_area(
                "prompt", key="pg_image_prompt", label_visibility="collapsed",
                placeholder="Describe an image to generate…", height=90,
            )
            go = st.button("Generate Image", type="primary", width="stretch")

        elif mode == VOICE:
            audio = st.audio_input("Record", key="pg_voice",
                                   label_visibility="collapsed")
            go = st.button("Transcribe", type="primary", width="stretch",
                           disabled=audio is None)

        else:  # COMPARE
            prompt = st.text_area(
                "prompt", key="pg_compare_prompt", label_visibility="collapsed",
                placeholder="Ask anything — both models will answer…", height=90,
            )
            go = st.button("Ask Both", type="primary", width="stretch")

    # --- Results (full width, below the input) -------------------------------
    if go and mode == COMPARE:
        _run_compare(prompt)
    elif go and mode == IMAGE:
        _run_image(prompt)
    elif go and mode == VOICE and audio is not None:
        _run_voice(audio)


def _run_compare(prompt: str | None) -> None:
    if not prompt or not prompt.strip():
        st.warning("Please enter a prompt first.")
        return
    col_gemini, col_claude = st.columns(2)
    with col_gemini:
        with st.spinner("Gemini thinking..."):
            result = ask_gemini(prompt)
        _render_response("🟦 Gemini", result)
    with col_claude:
        with st.spinner("Claude thinking..."):
            result = ask_claude(prompt)
        _render_response("🟧 Claude", result)


def _run_image(prompt: str | None) -> None:
    if not prompt or not prompt.strip():
        st.warning("Describe an image first.")
        return
    with st.spinner("Gemini is drawing..."):
        result = generate_image(prompt)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
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


def _run_voice(audio) -> None:
    with st.spinner("Gemini is listening..."):
        result = transcribe_audio(audio.getvalue(), mime_type="audio/wav")

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if result.ok:
            st.markdown(f"**Transcript:** {result.text}")
            st.caption(
                f"`{result.model}` · {result.latency_s}s · "
                f"{result.input_tokens} in / {result.output_tokens} out tokens"
            )
        else:
            st.error(result.error)
