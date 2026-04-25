from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from youtube_rag.config import Settings


def get_embedding_model(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
