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
    describe_image,
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
    """Gemini generates the image; Claude analyzes it with vision.

    Claude has no image-generation endpoint, so the honest comparison is
    Gemini (generate) vs Claude (see/critique) — each model's real strength.
    """
    if not prompt or not prompt.strip():
        st.warning("Describe an image first.")
        return

    col_gemini, col_claude = st.columns(2)

    with col_gemini:
        st.subheader("🟦 Gemini — generates")
        with st.spinner("Gemini is drawing..."):
            gen = generate_image(prompt)
        if gen.ok:
            st.image(gen.image_bytes, caption=prompt, width="stretch")
            st.caption(
                f"`{gen.model}` · {gen.latency_s}s · "
                f"{gen.input_tokens} in / {gen.output_tokens} out tokens"
            )
        else:
            st.error(gen.error)

    with col_claude:
        st.subheader("🟧 Claude — analyzes")
        if gen.ok and gen.image_bytes:
            with st.spinner("Claude is looking..."):
                analysis = describe_image(
                    gen.image_bytes,
                    gen.image_mime_type or "image/png",
                    question=(
                        f"This image was generated from the prompt: '{prompt}'. "
                        "Describe what you actually see and how well it matches."
                    ),
                )
            if analysis.ok:
                st.markdown(analysis.text)
                st.caption(
                    f"`{analysis.model}` · {analysis.latency_s}s · "
                    f"{analysis.input_tokens} in / {analysis.output_tokens} out tokens"
                )
            else:
                st.error(analysis.error)
            st.info(
                "Claude can't *generate* images — it has no image output. "
                "It can *see* them, so here it analyzes Gemini's result."
            )
        else:
            st.warning("No image to analyze.")


def _run_voice(audio) -> None:
    """Gemini transcribes; Claude cannot accept audio at all."""
    col_gemini, col_claude = st.columns(2)

    with col_gemini:
        st.subheader("🟦 Gemini — transcribes")
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

    with col_claude:
        st.subheader("🟧 Claude — not supported")
        st.info(
            "Claude's API does not accept audio input — it supports text, "
            "images, and PDFs, but not speech. Audio transcription is a "
            "Gemini-only capability."
        )
