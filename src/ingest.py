
from pathlib import Path

from pypdf import PdfReader

from src.chunking import chunk_text
from src.retrieval import client, embedding_function


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"


def ingest_documents():

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in the documents folder."
        )

    try:
        client.delete_collection(name="rag_documents")
        print("Old vector database collection deleted.")
    except Exception:
        print("Creating a new vector database collection.")

    collection = client.create_collection(
        name="rag_documents",
        embedding_function=embedding_function
    )

    total_chunks = 0

    for pdf_file in pdf_files:

        print(f"\nProcessing: {pdf_file.name}")

        reader = PdfReader(str(pdf_file))

        ids = []
        documents = []
        metadatas = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""
            chunks = chunk_text(text)

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                chunk_id = (
                    f"{pdf_file.stem}_page{page_number}_chunk{chunk_number}"
                )

                ids.append(chunk_id)
                documents.append(chunk)

                metadatas.append({
                    "source": pdf_file.name,
                    "page": page_number,
                    "chunk": chunk_number
                })

        if ids:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

        total_chunks += len(ids)

        print(f"Stored {len(ids)} chunks.")

    print("\nIngestion completed!")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total chunks in database: {collection.count()}")

    return collection.count()


if __name__ == "__main__":
    ingest_documents()
