from pathlib import Path

from .config import AppConfig, load_config
from .ingestion.chunking import chunk_pages
from .ingestion.documents import extract_pdfs
from .retrieval.bm25 import BM25Retriever


class RagSystem:
    def __init__(self, config: AppConfig | None = None, bm25_only: bool = False):
        self.config = config or load_config()
        self.pages = extract_pdfs(Path(self.config.paths.raw_documents))
        self.chunks = chunk_pages(
            self.pages,
            chunk_size=self.config.chunking.chunk_size,
            overlap=self.config.chunking.chunk_overlap,
        )
        self.bm25 = BM25Retriever(self.chunks)

        if bm25_only:
            return

        from google import genai

        from .generation.answer import answer_question
        from .retrieval.hybrid import HybridRetriever
        from .retrieval.vector import SentenceTransformerEmbedder, build_store

        self.embedder = SentenceTransformerEmbedder(self.config.embedding.model_name)
        self.store = build_store(self.chunks, self.embedder)
        self.hybrid_retriever = HybridRetriever(
            self.store,
            self.bm25,
            self.embedder,
            rrf_k=self.config.retrieval.rrf_k,
        )
        self.client = genai.Client()
        self._answer_question = answer_question

    def answer_query(self, query: str, use_gemini_llm: bool = True) -> dict:
        retrieved = self.hybrid_retriever.search(
            query,
            candidate_k=self.config.retrieval.candidate_k,
            top_k=self.config.retrieval.top_k,
        )
        result = self._answer_question(
            query,
            retrieved,
            client=self.client if use_gemini_llm else None,
            model_name=self.config.generation.model_name,
        )
        print(result["answer"])
        return result