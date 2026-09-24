import re

from rank_bm25 import BM25Okapi

from ..ingestion.chunking import Chunk


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized_chunks = [self._tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(tokenized_chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query: str, top_k: int) -> list[dict]:
        scores = self.index.get_scores(self._tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [
            {"chunk": self.chunks[index], "score": float(scores[index])}
            for index in ranked_indices[:top_k]
        ]