"""Streamlit app for RAG demo.

Start from project root with :
```bash
PYTHONPATH=. streamlit run demo/main.py
```
"""

import requests
import streamlit as st

from demo.ping import display_message_if_ping_fails

st.set_page_config(
    "RAG Example",
)

if "session" not in st.session_state:
    st.session_state.session = {}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"},
    ]


with st.sidebar:
    display_message_if_ping_fails()

st.title("RAG Example 🤖")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"},
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Which path gives me the candidate list?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = ""

    with st.spinner("Thinking..."):
        try:
            response = requests.post("http://localhost/chat", json={"query": prompt})
            response.raise_for_status()
            result = response.json()
            msg = result["message"]
        except Exception as e:
            st.error(e)
            st.stop()

    st.empty()

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
