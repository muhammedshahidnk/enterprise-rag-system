"""
STEP 3 + 4: Embeddings and Vector Database
-------------------------------------------

Embedding:
    SentenceTransformer("all-MiniLM-L6-v2")

Vector Database:
    FAISS IndexFlatIP

Because embeddings are normalized, inner product (IP) is equivalent
to cosine similarity.

Expected chunk object:
    chunk.text
    chunk.source
    chunk.page
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """
    Converts text into dense semantic embeddings using
    Sentence Transformers.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Convert a list of texts into normalized embeddings.

        Returns:
            np.ndarray of shape:
            (number_of_texts, embedding_dimension)
        """

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return np.asarray(vectors, dtype="float32")


class VectorStore:
    """
    FAISS vector database.

    Stores:
        - embedding vectors inside FAISS
        - original chunk objects separately

    This allows us to retrieve the original text and metadata
    after FAISS returns an index.
    """

    def __init__(self, dim: int):
        # Inner Product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dim)

        # Keep original chunks so FAISS index -> chunk mapping
        # can be recovered.
        self.chunks = []

    def add(self, vectors: np.ndarray, chunks: list):
        """
        Add vectors and their corresponding chunks.
        """

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ):
        """
        Search for the most similar chunks.

        Returns:
            [
                (chunk, similarity_score),
                ...
            ]
        """

        # FAISS expects:
        # (number_of_queries, embedding_dimension)

        query_vector = query_vector.reshape(1, -1)

        scores, indices = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            # -1 means no result
            if idx == -1:
                continue

            chunk = self.chunks[idx]

            results.append(
                (chunk, float(score))
            )

        return results


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_store(
    chunks,
    embedder: SentenceTransformerEmbedder
) -> VectorStore:
    # Extract text from each chunk
    texts = [chunk.text for chunk in chunks]
    # Generate embeddings
    vectors = embedder.encode(texts)
    # Embedding dimension
    dimension = vectors.shape[1]
    # Create FAISS database
    store = VectorStore(dim=dimension)
    # Add embeddings + chunks
    store.add(
        vectors,
        chunks
    )
    return store
