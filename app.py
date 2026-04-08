# Import sys to resolve an error message about os not defined
import sys
import os
import streamlit as st
# from openai import OpenAI
# Replace openai with gemini
import google.generativeai as genai
from retriever import retrieve_with_sources
import zipfile

# Use pure python to resolve compatibility issues with C++
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Use Streamlit secrets for API key
try:
    Gemini_API_Key = st.secrets["Gemini_API_Key"]
except Exception:
    st.error("Gemini API key is missing!")
    st.info('Create `.streamlit/secrets.toml` with: Gemini_API_KEY = "Al..."')
    st.stop()

genai.configure(api_key=Gemini_API_Key)

MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
model = genai.GenerativeModel(MODEL_NAME)

# Ensure ChromaDB exists
CHROMA_DIR = "./chroma_db"

# Unzip chroma_db
# if not os.path.exists(CHROMA_DIR):
#     with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
#         zip_ref.extractall(".")
if not os.path.exists(CHROMA_DIR):
    from ingest import ingest
    ingest()
# if not os.path.exists(CHROMA_DIR):
#     st.error("Knowledge base not found. Please run ingestion locally and upload chroma_db.")
#     st.stop()

# Prompt
SYSTEM_PROMPT = """
You are CareerAI, a warm, experienced, professional, and encouraging Data Science and Tech Career Advisor.
Answer ONLY based on the retrieved documents.
If the information is not in the context, say: "I don't have that specific information in my knowledge base."
Always be encouraging and give actionable advice.
"""

# Page configuration
st.set_page_config(page_title="Data Science Career Advisor", page_icon="🎯", layout="centered")

st.title("🎯 Data Science & Tech Career Advisor")
st.caption("Powered by 2020 to 2026 reports, roadmaps & interview guides")

# Quick buttons
st.markdown("**Quick questions:**")
example_prompts = [
    "What are the top skills for 2026?",
    "How do I build a strong data science portfolio?",
    "How do I prepare for a Data Science interview?",
    "What salary can I expect as a junior data scientist?",
    "Best roadmap to become a data scientist in 2026?",
]

cols = st.columns(3)
for i, prompt in enumerate(example_prompts):
    with cols[i % 3]:
        if st.button(prompt, use_container_width=True):
            st.session_state["queued_prompt"] = prompt

st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    label = src["url"].split("/")[-1] if src.get("type") == "pdf" else (src["url"][:70] + "...")
                    st.markdown(f"[{label}]({src['url']})")

# User input
user_input = st.chat_input("Ask your career question...")

if "queued_prompt" in st.session_state:
    user_input = st.session_state.pop("queued_prompt")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Searching knowledge base..."):
        context, sources = retrieve_with_sources(user_input, k=6)

    augmented_prompt = f"""
Use ONLY the following context from trusted documents:

CONTEXT:
{context}

QUESTION:
{user_input}
"""

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        response = model.generate_content(
            f"{SYSTEM_PROMPT}\n\n{augmented_prompt}",
            generation_config={
            "temperature": 0.4,
            "max_output_tokens": 900,
            }
        )

        full_response = response.text
        response_placeholder.markdown(full_response)
        
        if sources:
            with st.expander("📄 Sources used"):
                for src in sources:
                    label = src["url"].split("/")[-1] if src.get("type") == "pdf" else (src["url"][:70] + "...")
                    st.markdown(f"[{label}]({src['url']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources
    })
