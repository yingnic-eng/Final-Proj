"""
Transcript Processor — ProfessorGPT
Handles text cleaning, normalization, and (optionally) audio transcription.
"""

import re
import unicodedata


def process_transcript(raw: str) -> str:
    """
    Clean and normalize raw lecture transcript text.
    Handles pasted text, exported captions, and plain .txt files.
    """
    text = raw

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove common caption artifacts: [Music], (applause), timestamps
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?\s*", "", text)

    # Collapse multiple spaces/newlines
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)

    return text.strip()


def estimate_read_time(text: str) -> float:
    """Estimate read time in minutes (avg 200 wpm reading speed)."""
    words = len(text.split())
    return round(words / 200, 1)


# ── Optional: Audio transcription via Whisper API ─────────────────────────
def transcribe_audio(file_path: str, api_key: str) -> str:
    """
    Transcribe an audio/video file using OpenAI Whisper API.
    Requires: pip install openai
    Set OPENAI_API_KEY in your environment.

    Args:
        file_path: Path to .mp3, .mp4, .wav, .m4a file
        api_key:   OpenAI API key

    Returns:
        Transcribed text string
    """
    try:
        import openai

        openai.api_key = api_key
        with open(file_path, "rb") as f:
            result = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
            )
        return process_transcript(result)
    except ImportError:
        raise ImportError("Install openai: pip install openai")
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")
