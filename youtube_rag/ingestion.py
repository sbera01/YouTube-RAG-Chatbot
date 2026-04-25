from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi


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

    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id.strip(), languages=[language])
    except TranscriptsDisabled as exc:
        raise TranscriptFetchError(
            f"{language_name} captions are disabled for this video. "
            f"Please paste a YouTube link with {language_name} captions."
        ) from exc
    except Exception as exc:
        raise TranscriptFetchError(
            f"Could not fetch a {language_name} transcript for this video. "
            f"Please paste a YouTube link with {language_name} captions."
        ) from exc

    transcript = " ".join(
        chunk.text.strip() for chunk in transcript_list if getattr(chunk, "text", "").strip()
    )

    if not transcript:
        raise TranscriptFetchError(
            f"The {language_name} transcript is empty. "
            f"Please paste a YouTube link with spoken {language_name} content."
        )

    return transcript


def save_transcript(video_id: str, transcript: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{video_id}.txt"
    file_path.write_text(transcript, encoding="utf-8")
    return file_path
