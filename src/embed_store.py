"""
STEP 3 + 4: Embeddings and Vector Database
--------------------------------------------
Real path (use this on your own machine / server with internet access):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts, normalize_embeddings=True)

This sandbox cannot reach huggingface.co to download that model (network is
restricted here), so below we provide BOTH:
  1. `SentenceTransformerEmbedder` - the real thing, use this normally.
  2. `TfidfEmbedder` - a drop-in fallback with the exact same interface,
     used ONLY so this demo can run end-to-end right now without internet
     access to Hugging Face. Swap back to #1 in your own environment -
     the rest of the pipeline (FAISS, retrieval, prompting) doesn't change
     AT ALL. That's the point of separating embedding from retrieval.
     this is the entire work
"""
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


class SentenceTransformerEmbedder:
    """Real embedder. Requires internet access to huggingface.co the first time
    (it caches the model locally after that)."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype="float32")


class TfidfEmbedder:
    """Sandbox-only fallback so the pipeline runs without internet access.
    Same .encode() interface as the real embedder -> nothing downstream cares
    which one is plugged in."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2048)
        self._fitted = False

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        vecs = self.vectorizer.transform(texts).toarray().astype("float32")
        # normalize so we can use inner-product search like cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms


class VectorStore:
    """Thin wrapper around a FAISS index that keeps chunk objects alongside
    their vectors so we can map a search hit back to text + source + page."""
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine sim (vectors are normalized)
        self.chunks = []

    def add(self, vectors: np.ndarray, chunks: list):
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int):
        scores, idxs = self.index.search(query_vector.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


def build_store(chunks, embedder) -> VectorStore:
    texts = [c.text for c in chunks]
    if isinstance(embedder, TfidfEmbedder):
        embedder.fit(texts)
    vectors = embedder.encode(texts)
    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)
    return store
