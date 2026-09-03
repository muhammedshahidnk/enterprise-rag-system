"""
STEP 2: Chunking
-----------------
Why chunk at all? Embedding models and LLM context windows work better on
small, semantically coherent pieces of text than on whole pages/documents.
Too big -> retrieval gets "fuzzy" (a chunk half-matches many questions,
none exactly). Too small -> you lose context and answers get fragmented.

We chunk by character count with overlap, and we chunk PER PAGE (never
merging text across pages) so that "source page" stays a single, honest
number instead of a range.
"""
from dataclasses import dataclass
from extract import PageDoc


@dataclass
class Chunk:
    source: str
    page_num: int
    chunk_id: int
    text: str


def chunk_pages(pages: list[PageDoc], chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    chunks = []
    cid = 0
    for page in pages:
        text = page.text
        start = 0
        if len(text) <= chunk_size:
            chunks.append(Chunk(page.source, page.page_num, cid, text))
            cid += 1
            continue
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(page.source, page.page_num, cid, piece))
                cid += 1
            start += chunk_size - overlap  # move forward with overlap
    return chunks


if __name__ == "__main__":
    from src.extract import extract_pdfs
    pages = extract_pdfs("documents")
    for size in (300, 800):
        chunks = chunk_pages(pages, chunk_size=size, overlap=50)
        avg_len = sum(len(c.text) for c in chunks) / len(chunks)
        print(f"chunk_size={size}: {len(chunks)} chunks, avg length {avg_len:.0f} chars")
