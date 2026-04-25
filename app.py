from __future__ import annotations

import streamlit as st

from youtube_rag.llm_chain import build_rag_chain
from youtube_rag.config import get_settings
from youtube_rag.embeddings import get_embedding_model
from youtube_rag.ingestion import (
    TranscriptFetchError,
    extract_video_id,
    fetch_transcript,
    save_transcript,
)
from youtube_rag.retriever import build_or_load_vector_store, get_retriever

st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")

settings = get_settings()
language_name = "English" if settings.transcript_language.lower() == "en" else settings.transcript_language.upper()


@st.cache_resource
def load_embeddings():
    return get_embedding_model(settings)


def initialize_session_state() -> None:
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "active_video_link" not in st.session_state:
        st.session_state.active_video_link = ""
    if "active_video_id" not in st.session_state:
        st.session_state.active_video_id = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []


initialize_session_state()

st.title("YouTube RAG Chatbot")
st.caption(f"Ask grounded questions from YouTube videos with {language_name} transcripts.")

with st.sidebar:
    st.header("Index Video")
    st.caption(f"Paste a YouTube link with {language_name} spoken content and {language_name} captions.")
    video_link = st.text_input(
        "YouTube video link",
        value=st.session_state.active_video_link,
        placeholder="https://www.youtube.com/watch?v=...",
    )
    force_rebuild = st.checkbox("Rebuild database index", value=False)

    if st.button("Build / Load Index", type="primary"):
        if not video_link.strip():
            st.warning(f"Please paste a YouTube link with {language_name} content.")
        else:
            with st.spinner("Fetching transcript and preparing index..."):
                try:
                    video_id = extract_video_id(video_link, settings.transcript_language)
                    transcript = fetch_transcript(
                        video_id=video_id,
                        language=settings.transcript_language,
                    )
                    transcript_path = save_transcript(
                        video_id=video_id,
                        transcript=transcript,
                        output_dir=settings.transcript_root,
                    )

                    embeddings = load_embeddings()
                    vector_store = build_or_load_vector_store(
                        video_id=video_id,
                        transcript=transcript,
                        embeddings=embeddings,
                        settings=settings,
                        force_rebuild=force_rebuild,
                    )
                    retriever = get_retriever(vector_store, settings)

                    st.session_state.rag_chain = build_rag_chain(retriever, settings)
                    st.session_state.active_video_link = video_link.strip()
                    st.session_state.active_video_id = video_id
                    st.session_state.messages = []

                    st.success("Index is ready for chat.")
                    st.caption(f"Transcript saved at: {transcript_path}")
                except TranscriptFetchError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error("Unexpected error while preparing the index.")
                    st.exception(exc)

if st.session_state.rag_chain is None:
    st.info("Build or load an index from the sidebar to start chatting.")
else:
    st.write(f"Active video ID: {st.session_state.active_video_id}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the indexed video")

if question:
    if st.session_state.rag_chain is None:
        st.warning("Build an index first from the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    answer = st.session_state.rag_chain.invoke(question)
                except Exception as exc:
                    answer = "I could not generate an answer right now."
                    st.error("The model request failed.")
                    st.exception(exc)

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
