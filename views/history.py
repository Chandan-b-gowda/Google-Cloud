"""History page — list past conversations from Firestore, reload or delete them."""

import streamlit as st

from storage import delete_conversation, list_conversations


def render() -> None:
    st.header("Conversation History")

    conversations, query_ms = list_conversations(limit=25)
    st.caption(
        f"Loaded {len(conversations)} conversation(s) from Cloud Firestore "
        f"({query_ms} ms)"
    )

    if not conversations:
        st.info("No saved conversations yet. Start one in the **Chat** tab.")
        return

    for conv in conversations:
        header = f"**{conv.title}**  ·  `{conv.model}`  ·  {len(conv.messages)} messages"
        with st.expander(header):
            for msg in conv.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            load_col, del_col = st.columns(2)
            if load_col.button(
                "Load into Chat", key=f"load_{conv.id}", width='stretch'
            ):
                # Restore the conversation into the Chat page's session state.
                st.session_state.chat_messages = list(conv.messages)
                st.session_state.conv_id = conv.id
                st.session_state.chat_model_choice = (
                    "Gemini" if "gemini" in conv.model.lower() else "Claude"
                )
                # Programmatic navigation request (handled in app.py before the
                # nav widget is created, to avoid mutating a live widget key).
                st.session_state["_nav_to"] = "Chat"
                st.rerun()

            if del_col.button(
                "Delete", key=f"del_{conv.id}", width='stretch'
            ):
                _, delete_ms = delete_conversation(conv.id)
                st.toast(f"Deleted ({delete_ms} ms)")
                st.rerun()
