from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    embedding_model_name: str
    llm_repo_id: str
    llm_temperature: float
    chunk_size: int
    chunk_overlap: int
    top_k: int
    transcript_language: str
    index_root: Path
    transcript_root: Path


def get_settings() -> Settings:
    index_root = PROJECT_ROOT / "data" / "faiss"
    transcript_root = PROJECT_ROOT / "data" / "transcripts"

    index_root.mkdir(parents=True, exist_ok=True)
    transcript_root.mkdir(parents=True, exist_ok=True)

    return Settings(
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
        llm_repo_id=os.getenv("LLM_REPO_ID", "meta-llama/Llama-3.1-8B-Instruct"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("TOP_K", "4")),
        transcript_language=os.getenv("TRANSCRIPT_LANGUAGE", "en"),
        index_root=index_root,
        transcript_root=transcript_root,
    )
