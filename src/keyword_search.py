from rank_bm25 import BM25Okapi
import re


class BM25Retriever:
    """
    Keyword-based BM25 retriever.

    Uses the same chunks that are already used by FAISS.
    """

    def __init__(self, chunks):
        self.chunks = chunks

        # Tokenize every chunk
        self.tokenized_chunks = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    @staticmethod
    def _tokenize(text):
        """
        Simple tokenizer.

        Lowercase + split into words/numbers.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query, top_k=5):
        """
        Return top-k chunks ranked by BM25.
        """

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        # Sort highest score first
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for idx in ranked_indices[:top_k]:

            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx])
            })

        return results