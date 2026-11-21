
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Read the API key from Streamlit Cloud Secrets first.
# If unavailable, use the local .env file.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

if not API_KEY:
    st.error(
        "Gemini API key not found. "
        "Please configure GEMINI_API_KEY in your environment "
        "or Streamlit Cloud Secrets."
    )
    st.stop()

# Make the key available before importing generation.py.
os.environ["GEMINI_API_KEY"] = API_KEY

# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------

from src.retrieval import get_collection, retrieve_documents
from src.ingest import ingest_documents


# --------------------------------------------------
# INITIALIZE KNOWLEDGE BASE
# --------------------------------------------------

@st.cache_resource
def initialize_knowledge_base():

    collection = get_collection()

    # On Streamlit Cloud, the database starts empty.
    # Automatically process the PDFs on the first launch.
    if collection.count() == 0:
        ingest_documents()
        collection = get_collection()

    total_chunks = collection.count()

    if total_chunks == 0:
        raise RuntimeError(
            "No document chunks were found. "
            "Please check the documents folder."
        )

    return total_chunks


try:
    with st.spinner("Initializing document knowledge base..."):
        total_chunks = initialize_knowledge_base()

except Exception as error:
    st.error("Unable to initialize the document knowledge base.")
    st.exception(error)
    st.stop()


# Import generation only after configuring the API key.
from src.generation import generate_answer


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📚 RAG Document Assistant")

st.markdown(
    """
    Ask questions about your PDF documents.

    Get AI-generated answers grounded in your knowledge base,
    with source references.
    """
)

st.success(
    f"Knowledge base ready — {total_chunks} document chunks indexed."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ Project Information")

    st.success("RAG system initialized")

    st.markdown("### Technology Stack")
    st.write("**Embedding:** all-MiniLM-L6-v2")
    st.write("**Vector Database:** ChromaDB")
    st.write("**LLM:** Google Gemini")
    st.write("**Knowledge Base:** 4 PDF documents")
    st.write(f"**Indexed Chunks:** {total_chunks}")

    st.divider()

    st.markdown("### How it works")
    st.write("1. Ask a question.")
    st.write("2. Search relevant PDF chunks.")
    st.write("3. Send retrieved context to Gemini.")
    st.write("4. Receive an answer with sources.")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:
                with st.expander("📄 Retrieved Sources"):

                    for source in sources:
                        st.info(
                            f"**Source:** {source['source']}\n\n"
                            f"**Page:** {source['page']}"
                        )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask something about your PDF documents..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                retrieved = retrieve_documents(
                    question,
                    n_results=5
                )

                answer = generate_answer(question)

            is_error = answer.startswith(
                (
                    "Unable to generate",
                    "Configuration error",
                    "Gemini API error",
                    "Error:"
                )
            )

            if is_error:
                st.error(answer)
            else:
                st.markdown(answer)

            if retrieved:

                st.markdown("### 📄 Retrieved Sources")

                for source in retrieved:

                    st.info(
                        f"**Source:** {source['source']} | "
                        f"**Page:** {source['page']}"
                    )

            elif not is_error:
                st.warning(
                    "No document sources were retrieved."
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": retrieved,
                "is_error": is_error
            })

        except Exception as error:

            error_message = (
                "Something went wrong while processing "
                "your question. Please try again."
            )

            st.error(error_message)
            print(f"Application error: {error}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
                "is_error": True
            })


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "RAG Document Assistant | Powered by Python, "
    "ChromaDB, Sentence Transformers and Google Gemini"
)
