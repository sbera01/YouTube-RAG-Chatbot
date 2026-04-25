from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from youtube_rag.config import Settings


def _get_index_path(settings: Settings, video_id: str) -> Path:
	return settings.index_root / video_id


def build_or_load_vector_store(
	video_id: str,
	transcript: str,
	embeddings: HuggingFaceEmbeddings,
	settings: Settings,
	force_rebuild: bool = False,
) -> FAISS:
	index_path = _get_index_path(settings, video_id)

	if index_path.exists() and not force_rebuild:
		return FAISS.load_local(
			str(index_path),
			embeddings,
			allow_dangerous_deserialization=True,
		)

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=settings.chunk_size,
		chunk_overlap=settings.chunk_overlap,
	)
	docs = splitter.create_documents([transcript], metadatas=[{"video_id": video_id}])

	vector_store = FAISS.from_documents(docs, embeddings)
	index_path.mkdir(parents=True, exist_ok=True)
	vector_store.save_local(str(index_path))

	return vector_store


def get_retriever(vector_store: FAISS, settings: Settings):
	return vector_store.as_retriever(
		search_type="similarity",
		search_kwargs={"k": settings.top_k},
	)
