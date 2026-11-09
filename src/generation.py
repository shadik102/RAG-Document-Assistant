import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.retrieval import retrieve_documents

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """
You are a helpful document-based question-answering assistant.

Answer questions using only the supplied document context.
Do not invent facts or add unsupported information.
If the documents do not contain the answer, clearly say:
"The provided documents do not contain enough information to answer this question."

Mention the source document and page number when possible.
"""


def generate_answer(question):
    if not API_KEY:
        return "Error: GEMINI_API_KEY was not found in the .env file."

    retrieved = retrieve_documents(question)

    if not retrieved:
        return "No relevant information was found in the documents."

    context = ""

    for index, item in enumerate(retrieved, start=1):
        context += (
            f"\nDocument {index}\n"
            f"Source: {item['source']}\n"
            f"Page: {item['page']}\n"
            f"Content: {item['content']}\n"
        )

    prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer using the document context only.
Cite the source filename and page number.
"""

    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=30000)
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=700
            )
        )

        return response.text or "Gemini returned an empty response."

    except Exception as error:
        return f"Gemini API error: {error}"


if __name__ == "__main__":
    question = input("Ask a question about your PDFs: ")

    print("\nGenerating document-based answer...\n")
    answer = generate_answer(question)

    print("=" * 60)
    print("RAG ANSWER")
    print("=" * 60)
    print(answer)

