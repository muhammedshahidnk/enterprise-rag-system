from ..ingestion.chunking import Chunk


class HybridRetriever:
    def __init__(self, store, bm25, embedder, rrf_k: int = 60):
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.rrf_k = rrf_k

    def search(self, query: str, candidate_k: int, top_k: int) -> list[dict]:
        semantic = self.store.search(self.embedder.encode([query])[0], candidate_k)
        keyword = [(item["chunk"], item["score"]) for item in self.bm25.search(query, candidate_k)]
        return self._fuse(semantic, keyword)[:top_k]

    def _fuse(self, semantic, keyword) -> list[dict]:
        scores: dict[int, float] = {}
        data: dict[int, dict] = {}
        for rank, (chunk, score) in enumerate(semantic, start=1):
            key = id(chunk)
            scores[key] = scores.get(key, 0.0) + 1 / (self.rrf_k + rank)
            data[key] = {"chunk": chunk, "semantic_score": score, "bm25_score": None}
        for rank, (chunk, score) in enumerate(keyword, start=1):
            key = id(chunk)
            scores[key] = scores.get(key, 0.0) + 1 / (self.rrf_k + rank)
            data.setdefault(key, {"chunk": chunk, "semantic_score": None, "bm25_score": None})["bm25_score"] = score
        return [
            {**data[key], "rrf_score": score}
            for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]