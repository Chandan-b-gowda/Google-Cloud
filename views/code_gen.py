"""Code Generation page — one task prompt sent to both models, generated
code shown side by side, followed by a performance comparison table.

Reuses the existing ask_gemini/ask_claude calls (the same single-turn
text functions used on Compare) — code generation isn't a special API
mode, it's a prompt that asks for code. The metrics below come entirely
from the two LLMResponse objects already returned by that one call each;
no extra requests are made to build the table.
"""

import pandas as pd
import streamlit as st

from benchmark.suite import estimate_cost
from chatbots import LLMResponse, ask_claude, ask_gemini


def _code_prompt(task: str, language: str) -> str:
    """Wrap the user's task description into a code-generation instruction."""
    language_hint = f" in {language}" if language.strip() else ""
    return (
        f"Write code{language_hint} for the following task:\n\n"
        f"{task}\n\n"
        "Return the code in a single fenced code block with the correct "
        "language tag, followed by a brief explanation (2-4 sentences) of "
        "how it works. Do not omit the explanation."
    )


def _render_response(title: str, result: LLMResponse) -> None:
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


def _build_comparison_table(
    gemini: LLMResponse, claude: LLMResponse
) -> pd.DataFrame:
    """Turn the two raw LLMResponses into a side-by-side metrics table.

    Every value here is read directly off the response objects already
    returned by the single ask_gemini/ask_claude call above — nothing is
    re-measured or re-requested.
    """
    rows = []
    for label, result in [("Gemini", gemini), ("Claude", claude)]:
        cost = estimate_cost(
            result.model, result.input_tokens or 0, result.output_tokens or 0
        )
        rows.append(
            {
                "Model": f"{label} (`{result.model}`)",
                "Latency (s)": result.latency_s,
                "Input tokens": result.input_tokens,
                "Output tokens": result.output_tokens,
                "Total tokens": (result.input_tokens or 0) + (result.output_tokens or 0),
                "Est. cost (USD)": cost,
                "Response length (chars)": len(result.text),
            }
        )
    return pd.DataFrame(rows)


def _render_reasoning(gemini: LLMResponse, claude: LLMResponse) -> None:
    """Plain-language summary of what the numbers above mean."""
    if not (gemini.ok and claude.ok):
        st.info("One model failed to respond — comparison reasoning needs both answers.")
        return

    faster = "Gemini" if (gemini.latency_s or 0) < (claude.latency_s or 0) else "Claude"
    diff = abs((gemini.latency_s or 0) - (claude.latency_s or 0))

    g_cost = estimate_cost(gemini.model, gemini.input_tokens or 0, gemini.output_tokens or 0)
    c_cost = estimate_cost(claude.model, claude.input_tokens or 0, claude.output_tokens or 0)
    cost_line = ""
    if g_cost is not None and c_cost is not None:
        cheaper = "Gemini" if g_cost < c_cost else "Claude"
        cost_line = f" {cheaper} was the cheaper call for this prompt."

    longer = "Gemini" if len(gemini.text) > len(claude.text) else "Claude"

    st.markdown(
        f"- **Speed:** {faster} responded faster by **{diff:.2f}s** for this prompt.\n"
        f"- **Cost:** estimated from list price per token.{cost_line}\n"
        f"- **Verbosity:** {longer} produced the longer response "
        f"({len(gemini.text)} vs {len(claude.text)} characters) — longer "
        "isn't necessarily better; check the code itself for correctness "
        "and style.\n"
        "- **Note:** a single run is not a reliable benchmark — latency in "
        "particular varies call to call. Use the Benchmark page to run "
        "multiple prompts and average the results before drawing conclusions."
    )


def render() -> None:
    st.markdown(
        "<h1 style='text-align:center;margin-bottom:0;'>Code Generation</h1>"
        "<p style='text-align:center;color:#9aa0a6;margin-top:4px;'>"
        "Describe a coding task — Gemini and Claude each generate code, "
        "compared side by side.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    _, center, _ = st.columns([1, 2, 1])
    with center:
        task = st.text_area(
            "task", key="code_task", label_visibility="collapsed",
            placeholder="Describe what the code should do…",
            value="Write a function that checks if a string is a palindrome.",
            height=100,
        )
        language = st.text_input(
            "Language (optional — leave blank to let the model choose)",
            value="Python", key="code_lang",
        )
        go = st.button("Generate Code", type="primary", width="stretch")

    if not go:
        return
    if not task.strip():
        st.warning("Please describe the task first.")
        return

    prompt = _code_prompt(task, language)

    col_gemini, col_claude = st.columns(2)
    with col_gemini:
        with st.spinner("Gemini is coding..."):
            gemini_result = ask_gemini(prompt)
        _render_response("Gemini", gemini_result)
    with col_claude:
        with st.spinner("Claude is coding..."):
            claude_result = ask_claude(prompt)
        _render_response("Claude", claude_result)

    st.divider()
    st.subheader("Performance Comparison")
    st.dataframe(
        _build_comparison_table(gemini_result, claude_result),
        width="stretch", hide_index=True,
    )
    _render_reasoning(gemini_result, claude_result)