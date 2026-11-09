from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

from chunking import chunk_text

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="rag_documents",
    embedding_function=embedding_function
)


def ingest_documents():
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the documents folder.")
        return

    total_chunks = 0

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        reader = PdfReader(str(pdf_file))

        ids = []
        documents = []
        metadatas = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            chunks = chunk_text(text)

            for chunk_number, chunk in enumerate(chunks, start=1):
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
        print(f"Stored {len(ids)} chunks from {pdf_file.name}")

    print(f"\nIngestion completed! Total chunks: {total_chunks}")
    print(f"Total chunks in database: {collection.count()}")


if __name__ == "__main__":
    ingest_documents()
