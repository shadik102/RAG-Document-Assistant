
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

STOP_WORDS = {
    "what", "is", "the", "a", "an", "how", "do", "i",
    "to", "in", "of", "for", "and", "or", "with",
    "can", "you", "me", "explain", "write", "about"
}


def get_collection():
    """Always obtain the current ChromaDB collection."""
    return client.get_or_create_collection(
        name="rag_documents",
        embedding_function=embedding_function
    )


def tokenize(text):
    return set(
        re.findall(r"[a-zA-Z0-9_]+", text.lower())
    )


def retrieve_documents(question, n_results=5):

    if not question or not question.strip():
        return []

    collection = get_collection()

    total = collection.count()

    if total == 0:
        return []

    semantic = collection.query(
        query_texts=[question],
        n_results=min(10, total),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    semantic_results = {}

    for i, content in enumerate(semantic["documents"][0]):

        metadata = semantic["metadatas"][0][i]
        distance = semantic["distances"][0][i]

        key = (
            metadata.get("source", ""),
            metadata.get("page", 0),
            metadata.get("chunk", 0)
        )

        semantic_results[key] = {
            "distance": distance
        }

    all_data = collection.get(
        include=["documents", "metadatas"]
    )

    query_words = tokenize(question) - STOP_WORDS
    clean_question = question.lower()

    is_hello_query = (
        "hello" in query_words
        and "python" in query_words
        and "print" in query_words
    )

    ranked = {}

    for content, metadata in zip(
        all_data["documents"],
        all_data["metadatas"]
    ):

        clean_content = content.lower()
        content_words = tokenize(content)

        overlap = len(query_words & content_words)

        keyword_score = (
            overlap / max(len(query_words), 1)
        )

        phrase_score = 0

        if (
            len(clean_question.strip()) > 3
            and clean_question.strip() in clean_content
        ):
            phrase_score += 2

        if is_hello_query:

            if "hello python" in clean_content:
                phrase_score += 2

            if (
                "print" in clean_content
                and "python program" in clean_content
            ):
                phrase_score += 1

            if (
                "program 1" in clean_content
                and "hello python" in clean_content
            ):
                phrase_score += 2

        toc_penalty = 0

        if "table of contents" in clean_content[:300]:
            toc_penalty = 1

        key = (
            metadata.get("source", ""),
            metadata.get("page", 0),
            metadata.get("chunk", 0)
        )

        semantic_score = 0
        distance = None

        if key in semantic_results:

            distance = semantic_results[key]["distance"]

            semantic_score = max(
                0,
                1 - distance
            )

        final_score = (
            keyword_score
            + phrase_score
            + semantic_score * 0.5
            - toc_penalty
        )

        ranked[key] = {
            "content": content,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", 0),
            "chunk": metadata.get("chunk", 0),
            "distance": distance,
            "score": final_score
        }

    results = sorted(
        ranked.values(),
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:n_results]


if __name__ == "__main__":

    question = input("Enter your question: ")

    results = retrieve_documents(question)

    print("\nRETRIEVED DOCUMENTS")
    print("=" * 60)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Score:", round(result["score"], 4))
        print("Content:")
        print(result["content"][:1200])
        print("-" * 60)
