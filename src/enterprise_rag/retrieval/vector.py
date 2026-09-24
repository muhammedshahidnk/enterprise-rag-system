import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ..ingestion.chunking import Chunk


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        scores, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        return [
            (self.chunks[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index != -1
        ]


def build_store(chunks: list[Chunk], embedder: SentenceTransformerEmbedder) -> VectorStore:
    vectors = embedder.encode([chunk.text for chunk in chunks])
    store = VectorStore(vectors.shape[1])
    store.add(vectors, chunks)
    return store