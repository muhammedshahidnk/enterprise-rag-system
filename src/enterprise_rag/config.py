from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathConfig:
    raw_documents: Path
    evaluation_dataset: Path


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    candidate_k: int
    rrf_k: int


@dataclass(frozen=True)
class GenerationConfig:
    provider: str
    model_name: str
    temperature: float
    max_output_tokens: int


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    paths: PathConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_config(config_path: str | Path | None = None) -> AppConfig:
    configured_path = config_path or os.getenv("ENTERPRISE_RAG_CONFIG")
    path = Path(configured_path) if configured_path else Path.cwd() / "Config" / "config.yaml"
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / "Config" / "config.yaml"
    project_root = path.parent.parent

    with path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}

    paths = raw.get("paths", {})
    chunking = raw.get("chunking", {})
    embedding = raw.get("embedding", {})
    retrieval = raw.get("retrieval", {})
    hybrid = raw.get("hybrid_retrieval", {})
    generation = raw.get("generation", {})

    return AppConfig(
        project_root=project_root,
        paths=PathConfig(
            raw_documents=_resolve_path(project_root, paths["raw_documents"]),
            evaluation_dataset=_resolve_path(project_root, paths["evaluation_dataset"]),
        ),
        chunking=ChunkingConfig(
            chunk_size=int(chunking["chunk_size"]),
            chunk_overlap=int(chunking["chunk_overlap"]),
        ),
        embedding=EmbeddingConfig(
            model_name=str(embedding.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        ),
        retrieval=RetrievalConfig(
            top_k=int(retrieval["top_k"]),
            candidate_k=int(retrieval.get("candidate_k", retrieval["top_k"])),
            rrf_k=int(hybrid.get("rrf_k", 60)),
        ),
        generation=GenerationConfig(
            provider=str(generation.get("provider", "gemini")),
            model_name=str(generation.get("model_name", "gemini-2.5-flash")),
            temperature=float(generation.get("temperature", 0.0)),
            max_output_tokens=int(generation.get("max_output_tokens", 500)),
        ),
    )