from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from langchain_community.document_loaders import YoutubeLoader


class TranscriptFetchError(RuntimeError):
    """Raised when the transcript cannot be fetched from YouTube."""


YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _is_valid_video_id(value: str) -> bool:
    return bool(YOUTUBE_ID_PATTERN.fullmatch(value))


def _language_display_name(language: str) -> str:
    code = language.strip().lower()
    if code == "en":
        return "English"
    return code.upper() if code else "configured"


def _fetch_with_langchain_loader(video_id: str, language_candidates: list[str]) -> str:
    loader = YoutubeLoader(video_id=video_id, language=language_candidates)
    docs = loader.load()
    transcript = " ".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())
    return transcript


def extract_video_id(video_url_or_id: str, language: str = "en") -> str:
    raw_value = video_url_or_id.strip()
    language_name = _language_display_name(language)

    if not raw_value:
        raise TranscriptFetchError(
            f"YouTube video link is required. Please paste a link for {language_name} content."
        )

    if _is_valid_video_id(raw_value):
        return raw_value

    parsed = urlparse(raw_value)
    host = parsed.netloc.lower().replace("www.", "")

    if host == "youtu.be":
        short_id = parsed.path.strip("/").split("/")[0] if parsed.path.strip("/") else ""
        if _is_valid_video_id(short_id):
            return short_id

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if _is_valid_video_id(query_id):
            return query_id

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
            if _is_valid_video_id(parts[1]):
                return parts[1]

    raise TranscriptFetchError(
        f"Could not extract a valid video ID from the provided YouTube link. "
        f"Please paste a valid YouTube link for {language_name} content."
    )


def fetch_transcript(video_id: str, language: str = "en") -> str:
    language_name = _language_display_name(language)

    if not video_id.strip():
        raise TranscriptFetchError("Video ID is required.")

    language_candidates = [language]
    if language.strip().lower() == "en":
        # Some videos expose region-specific English subtitle codes only.
        language_candidates.extend(["en-US", "en-GB", "en-IN", "en-CA", "en-AU"])

    try:
        transcript = _fetch_with_langchain_loader(video_id.strip(), language_candidates)
    except ImportError as exc:
        raise TranscriptFetchError(
            "Missing dependency for transcript loading. Install youtube-transcript-api and retry."
        ) from exc
    except Exception as exc:
        root_cause = str(exc).strip() or exc.__class__.__name__
        raise TranscriptFetchError(
            f"Could not fetch an {language_name} transcript for this video. "
            f"Please paste a YouTube link with {language_name} captions. "
            f"Root cause: {root_cause}. "
            f"If this works locally but fails on Streamlit Cloud, the platform IP may be rate-limited or blocked by YouTube."
        ) from exc

    if not transcript:
        raise TranscriptFetchError(
            f"No {language_name} transcript was found for this video. "
            f"Please paste a YouTube link with {language_name} captions."
        )

    return transcript


def save_transcript(video_id: str, transcript: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{video_id}.txt"
    file_path.write_text(transcript, encoding="utf-8")
    return file_path
