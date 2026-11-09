from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_collection(
    name="rag_documents",
    embedding_function=embedding_function
)


def retrieve_documents(question, n_results=3):
    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count())
    )

    retrieved_documents = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved_documents.append({
            "content": document,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance
        })

    return retrieved_documents


if __name__ == "__main__":
    question = input("Enter your question: ")

    results = retrieve_documents(question)

    print("\nRETRIEVED DOCUMENTS")
    print("=" * 60)

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Content:\n{result['content'][:1000]}")
        print("-" * 60)
