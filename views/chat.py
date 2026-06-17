"""Chat page — multi-turn conversation with one model, persisted to Firestore.

Conversation state lives in st.session_state across reruns:
    chat_messages : list[{role, content}]  — the running transcript
    conv_id       : str | None             — Firestore document id (None = unsaved)
"""

import streamlit as st

from chatbots import CHAT_FUNCTIONS
from storage import Conversation, save_conversation


def _init_state() -> None:
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("conv_id", None)


def render() -> None:
    _init_state()

    st.header("Multi-Turn Chat")

    top_left, top_right = st.columns([3, 1])
    with top_left:
        model_choice = st.radio(
            "Model",
            options=list(CHAT_FUNCTIONS.keys()),  # ["Gemini", "Claude"]
            horizontal=True,
            key="chat_model_choice",
            help="The model remembers everything earlier in this conversation.",
        )
    with top_right:
        st.write("")  # vertical spacer to align the button
        if st.button("New chat", width='stretch'):
            st.session_state.chat_messages = []
            st.session_state.conv_id = None
            st.rerun()

    if st.session_state.conv_id:
        st.caption(f"Firestore document: `{st.session_state.conv_id}`")

    # Replay the existing transcript as chat bubbles
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user message
    prompt = st.chat_input(f"Message {model_choice}...")
    if not prompt:
        return

    # 1. Show + record the user's message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the selected model with the FULL history (that's the "memory")
    chat_fn = CHAT_FUNCTIONS[model_choice]
    with st.chat_message("assistant"):
        with st.spinner(f"{model_choice} thinking..."):
            result = chat_fn(st.session_state.chat_messages)

        if not result.ok:
            st.error(result.error)
            # Drop the user turn we optimistically added so retry is clean
            st.session_state.chat_messages.pop()
            return

        st.markdown(result.text)
        st.caption(
            f"`{result.model}` · {result.latency_s}s · "
            f"{result.input_tokens} in / {result.output_tokens} out tokens"
        )
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": result.text}
        )

    # 3. Persist the whole conversation to Firestore (create or update)
    conv = Conversation(
        id=st.session_state.conv_id,
        title=st.session_state.chat_messages[0]["content"][:60],
        model=result.model,
        provider=result.provider,
        messages=st.session_state.chat_messages,
    )
    conv_id, write_ms = save_conversation(conv)
    st.session_state.conv_id = conv_id
    st.caption(f"Saved to Cloud Firestore ({write_ms} ms)")
