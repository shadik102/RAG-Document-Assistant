
import re


def chunk_text(text, chunk_size=250, overlap=50):
    """
    Split text into smaller overlapping word-based chunks.
    Default: 250 words per chunk with 50 words overlap.
    """

    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Clean unnecessary whitespace
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks
