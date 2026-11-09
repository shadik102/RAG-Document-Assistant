import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai

from src.generation import generate_answer
from src.retrieval import retrieve_documents

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Assistant")
st.write("Ask questions and get answers grounded in your PDF documents.")

if not API_KEY:
    st.error("Gemini API key not found. Check your .env file.")
    st.stop()

question = st.text_input(
    "Enter your question",
    placeholder="What is a reusable machine learning pipeline?"
)

if st.button("Compare RAG vs No RAG", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answers..."):
            rag_answer = generate_answer(question)

            try:
                client = genai.Client(api_key=API_KEY)

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=(
                        "Answer the following question using your general "
                        "knowledge. Do not access external documents.\n\n"
                        f"Question: {question}"
                    )
                )

                no_rag_answer = response.text or "No response received."

            except Exception as error:
                no_rag_answer = f"Gemini API error: {error}"

            retrieved = retrieve_documents(question)

        st.subheader("Answer comparison")

        tab1, tab2 = st.tabs(["RAG Answer", "No-RAG Answer"])

        with tab1:
            st.markdown("### Answer using your PDFs")
            st.write(rag_answer)

            st.markdown("### Retrieved sources")

            for item in retrieved:
                st.info(
                    f"Source: {item['source']} | Page: {item['page']}"
                )

        with tab2:
            st.markdown("### Normal Gemini answer")
            st.write(no_rag_answer)

st.sidebar.title("Project Information")
st.sidebar.write("**Embedding Model:** all-MiniLM-L6-v2")
st.sidebar.write("**Vector Database:** ChromaDB")
st.sidebar.write("**LLM:** Google Gemini")
st.sidebar.write("**Source Documents:** 4 PDFs")

st.sidebar.success("RAG system initialized")
