from dataclasses import dataclass

from .documents import PageDocument


@dataclass(frozen=True)
class Chunk:
    source: str
    page_num: int
    chunk_id: int
    text: str


def chunk_pages(pages: list[PageDocument], chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be between 0 and chunk_size")

    chunks: list[Chunk] = []
    chunk_id = 0
    step = chunk_size - overlap

    for page in pages:
        for start in range(0, len(page.text), step):
            text = page.text[start:start + chunk_size].strip()
            if text:
                chunks.append(Chunk(page.source, page.page_num, chunk_id, text))
                chunk_id += 1

    return chunks