"""Benchmark page — run the preset suite and show an auto-generated
comparison table of latency, tokens, and cost across both models.
"""

import pandas as pd
import streamlit as st

from benchmark import PRESET_PROMPTS, run_benchmark

# The qualitative dimensions from the assignment brief that can't be measured
# by running prompts — listed here as a checklist for the written report.
REPORT_DIMENSIONS = [
    ("Performance", "Measured here (latency + tokens)"),
    ("Response time", "Measured here (latency)"),
    ("Pricing", "Measured here (estimated cost) — verify with official pricing"),
    ("Accuracy", "Review the answers below for each category"),
    ("Scalability", "Research: rate limits, quotas, infrastructure"),
    ("Customization", "Research + your fine-tuning work on Vertex AI"),
    ("Ecosystem integration", "Vertex AI vs Anthropic API (first-hand)"),
    ("Ethics & safety", "Research: safety policies, refusals"),
    ("Ease of use", "First-hand: SDK and setup experience"),
    ("Language support", "See the Multilingual row below"),
    ("Data privacy & security", "First-hand: API key vs ADC/OAuth"),
    ("Community support", "Research: docs, forums, ecosystem size"),
    ("Use-case versatility", "First-hand: text, image, audio (Playground page)"),
]


def render() -> None:
    st.header("Benchmark Suite")
    st.markdown(
        "Runs the same set of prompts through both models and builds a "
        "comparison table of speed, token usage, and estimated cost."
    )

    if st.button("Run Benchmark", type="primary"):
        bar = st.progress(0.0, text="Starting...")

        def _progress(done: int, total: int) -> None:
            bar.progress(done / total, text=f"Running call {done} of {total}...")

        st.session_state.benchmark_rows = run_benchmark(on_progress=_progress)
        bar.empty()

    rows = st.session_state.get("benchmark_rows")
    if not rows:
        st.info("Click **Run Benchmark** to generate the comparison table.")
        _render_dimension_checklist()
        return

    df = pd.DataFrame([row.__dict__ for row in rows])

    # --- Summary: one row per model -----------------------------------------
    st.subheader("Summary by model")
    summary = (
        df.groupby("provider")
        .agg(
            avg_latency_s=("latency_s", "mean"),
            total_input_tokens=("input_tokens", "sum"),
            total_output_tokens=("output_tokens", "sum"),
            total_cost_usd=("cost_usd", "sum"),
        )
        .round(4)
    )
    st.dataframe(summary, width='stretch')

    # --- Detailed: one row per prompt per model -----------------------------
    st.subheader("Detailed results")
    detail = df[
        ["category", "provider", "model", "latency_s",
         "input_tokens", "output_tokens", "cost_usd"]
    ]
    st.dataframe(detail, width='stretch', hide_index=True)

    st.download_button(
        "Download results as CSV",
        data=df.to_csv(index=False),
        file_name="benchmark_results.csv",
        mime="text/csv",
    )

    # --- The actual answers, for the accuracy/quality comparison ------------
    with st.expander("See the full answers from each model"):
        for category, _ in PRESET_PROMPTS:
            st.markdown(f"#### {category}")
            for row in [r for r in rows if r.category == category]:
                st.markdown(f"**{row.provider}** · `{row.model}`")
                st.markdown(row.answer if row.ok else f"_error: {row.answer}_")
                st.caption(
                    f"{row.latency_s}s · {row.input_tokens} in / "
                    f"{row.output_tokens} out tokens"
                )
            st.markdown("---")

    _render_dimension_checklist()


def _render_dimension_checklist() -> None:
    """Map the assignment's comparison dimensions to where each is covered."""
    with st.expander("Comparison dimensions"):
        st.dataframe(
            pd.DataFrame(REPORT_DIMENSIONS, columns=["Dimension", "Covered by"]),
            width='stretch',
            hide_index=True,
        )
