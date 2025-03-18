import logging
from datetime import datetime
from json import dumps, load

import anthropic

# import boto3
import PyPDF2
import streamlit as st

# from botocore.exceptions import ClientError
from jinja2 import Environment, FileSystemLoader

# -------------------------------------------------------------------
# Setup Bedrock client, model IDs, and default inference config
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
client = anthropic.Anthropic()

model_ids = {
    "claude-3.7-sonnet": "claude-3-7-sonnet-latest",
    "claude-3.5-sonnet": "claude-3-5-sonnet-latest",
    "claude-3.5-haiku": "claude-3-5-haiku-latest",
}
INFERENCE_CONFIG = {"maxTokens": 1024, "temperature": 0.5, "topP": 0.9}

AVATARS = {
    "user": ":material/face:",
    "assistant": ":material/smart_toy:",
    "tool": ":material/build:",
}
# -------------------------------------------------------------------
# Initialize session state
# -------------------------------------------------------------------
if "full_conversation" not in st.session_state:
    st.session_state.full_conversation = []
if "visible_conversation" not in st.session_state:
    st.session_state.visible_conversation = []
if "file_context" not in st.session_state:
    st.session_state.file_context = []
if "file_injected" not in st.session_state:
    st.session_state.file_injected = False
if "model" not in st.session_state:
    st.session_state.model = "claude-3.5-sonnet"
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def clear_chat():
    st.session_state.full_conversation = []
    st.session_state.visible_conversation = []
    st.session_state.file_context = []
    st.session_state.file_injected = False


def read_file(file):
    if file:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        elif file.type in ("text/plain", "text/markdown") or (
            file.type == "application/octet-stream"
            and (file.name.endswith(".txt") or file.name.endswith(".md"))
        ):
            return file.read().decode("utf-8")
        elif file.type == "application/json":
            return dumps(load(file))
    return ""


def inject_file_context():
    """
    Adds uploaded file text to the hidden conversation exactly once.
    """
    if st.session_state.file_context and not st.session_state.file_injected:
        combined_context = "\n".join(st.session_state.file_context)
        hidden_message = {
            "role": "user",
            "content": [{"text": f"[Uploaded file context]\n{combined_context}"}],
        }
        st.session_state.full_conversation.append(hidden_message)
        st.session_state.file_injected = True


def load_prompt():
    env = Environment(loader=FileSystemLoader("prompts"))
    template = env.get_template("system.md")

    # Pick the right "extras" file
    match st.session_state.model:
        case "claude-3.5-sonnet":
            extras = env.get_template("3-5-sonnet_extras.md")
        case "claude-3.5-haiku":
            extras = env.get_template("3-5-haiku_extras.md")
        case "claude-3.7-sonnet":
            extras = env.get_template("3-7-sonnet_extras.md")
        case _:
            extras = env.from_string("")

    current_date = datetime.now().strftime("%B %d, %Y")
    return template.render(extras=extras.render(), current_date=current_date)


def get_system_prompt(style, custom_text):
    """
    Incorporate the user's style choice into the system prompt.
    """
    base_prompt = load_prompt()
    if style == "custom":
        text = custom_text.strip() if custom_text.strip() else ""
    elif style == "normal":
        text = ""
    else:
        text = "Always respond concisely."
    return f"{base_prompt}\n\n{text}" if text else base_prompt


def send_message_streaming(message):
    """
    Sends the user's message to Bedrock in streaming mode, and updates
    the Streamlit UI chunk by chunk as tokens arrive.
    """
    inject_file_context()

    # Record the user's message to full_conversation (the user's message is already in visible_conversation)
    user_msg = {"role": "user", "content": message}
    st.session_state.full_conversation.append(user_msg)

    # Build the system prompt
    system_prompt = get_system_prompt(style_option, custom_style)

    # Merge with the default config
    inference_config = INFERENCE_CONFIG | {
        "temperature": temperature,
        "maxTokens": max_tokens,
    }
    # If you want to pass extra fields (like 'top_k') to the model:
    thinking = (
        {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        if extended_thinking
        else {"type": "disabled"}
    )

    try:
        # TODO: ideally also display thinking blocks in the response
        # not sure how to do it yet though
        with st.spinner("Thinking..."):
            with client.messages.stream(
                model=model_ids[st.session_state.model],
                messages=st.session_state.full_conversation,
                system=system_prompt,
                max_tokens=inference_config["maxTokens"],
                temperature=inference_config["temperature"],
                thinking=thinking,
            ) as stream:
                content = st.write_stream(stream.text_stream)

        # After streaming is complete, store the full response in conversation history
        assistant_msg = {"role": "assistant", "content": content}
        st.session_state.full_conversation.append(assistant_msg)
        st.session_state.visible_conversation.append(
            {"role": "assistant", "content": content}
        )

    except Exception as e:
        st.error(f"Error: {e}")


def show_message(message):
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        if message["content"]:
            if message["role"] == "user":
                st.markdown(message["content"], unsafe_allow_html=True)
            elif message["role"] == "assistant":
                st.markdown(message["content"], unsafe_allow_html=True)


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.session_state.model = st.sidebar.selectbox(
    "Select model", list(model_ids.keys()), index=1
)


extended_thinking = False
if st.session_state.model == "claude-3.7-sonnet":
    extended_thinking = st.sidebar.checkbox("Enable extended thinking", value=False)

if extended_thinking:
    thinking_budget = st.sidebar.number_input(
        "Thinking budget (tokens)", 4000, 64000, 4000, 100
    )
else:
    thinking_budget = 0

max_tokens_limit = 64000 if extended_thinking else 8000
max_tokens = st.sidebar.number_input(
    "Max tokens",
    thinking_budget + 500,
    max_tokens_limit + 500,
    thinking_budget + 1000,
    100,
)
if not extended_thinking:
    temperature = st.sidebar.number_input("Temperature", 0.0, 1.0, 0.5)
else:
    temperature = 1.0

st.title("Chat with Claude")

if st.button("New Chat"):
    clear_chat()
    st.rerun()

style_option = st.sidebar.selectbox(
    "Select response style", ["normal", "concise", "custom"], index=1
)
custom_style = ""
if style_option == "custom":
    custom_style = st.sidebar.text_input(
        "Specify custom style",
        placeholder="e.g., Always respond in a witty and informal manner.",
    )

uploaded_files = st.sidebar.file_uploader(
    "Upload files to context",
    type=["txt", "pdf", "json", "md"],
    accept_multiple_files=True,
)
if uploaded_files:
    for file in uploaded_files:
        content = read_file(file)
        if content:
            st.session_state.file_context.append(content)
    st.sidebar.success("File(s) uploaded. Their content is now available as context.")

# Display conversation history (all previously completed messages)
for entry in st.session_state.visible_conversation:
    if entry["role"] == "user":
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(entry["content"])
    else:
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.markdown(entry["content"])

# Chat input at the bottom (similar to example.py)
if user_input := st.chat_input("Message Claude..."):
    # When the user sends a new message, add it to conversation history
    st.session_state.visible_conversation.append(
        {"role": "user", "content": user_input}
    )

    # Display the user's message
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)

    # Now show the assistant's response in a streaming fashion
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        send_message_streaming(user_input)

# Add a download button for the visible conversation
conversation_text = "\n\n".join(
    [
        f"**{entry['role'].capitalize()}**: {entry['content']}"
        for entry in st.session_state.visible_conversation
    ]
)
date = datetime.now().strftime("%Y%m%d")
st.sidebar.download_button(
    "Download conversation",
    data=conversation_text,
    file_name=f"conversation-{date}.md",
    mime="text/markdown",
    icon=":material/download:",
)
