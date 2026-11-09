def chunk_text(text, chunk_size=450, overlap=75):
    """
    Split text into overlapping word-based chunks.
    Approximately 450 words per chunk, with 75 words of overlap.
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks
