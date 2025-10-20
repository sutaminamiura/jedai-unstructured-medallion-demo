import logging
import os
import re

from databricks.sdk import WorkspaceClient
import streamlit as st

from knowledge_agent import create_knowlege_agent, create_assistant_message

from io import BytesIO
from PIL import Image


w = WorkspaceClient()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("🧱 Databricks Knowledge Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = create_knowlege_agent()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("何を調べますか？"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Query the Databricks serving endpoint
        assistant_response = st.session_state.agent.invoke(
            input={"messages": {"role": "user", "content": prompt}}
        )["messages"]
        assistant_response = create_assistant_message(assistant_response)

        st.markdown(assistant_response["text"])

        # 画像の表示
        images_for_disp = []
        for image_file in assistant_response["images"]:
            response = w.files.download(image_file)
            file_data = response.contents.read()
            images_for_disp.append(file_data)

        st.image(images_for_disp)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response["text"]}
    )
