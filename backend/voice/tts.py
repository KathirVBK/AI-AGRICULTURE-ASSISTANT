"""
AgriSense-AI — voice/tts.py
Text-to-Speech via gpt-4o-mini-tts (Navigate Labs API).
Supports English and Tamil. Uses streaming for low-latency response.
"""

import io
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Singleton client — initialized once, reused on every request ──
_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("NAVIGATE_API_KEY"),
            base_url=os.getenv("NAVIGATE_BASE_URL", "https://apidev.navigatelabsai.com/v1")
        )
    return _client


def _truncate_for_speech(text: str, max_chars: int = 1200) -> str:
    """
    Truncate text at a natural sentence boundary to keep TTS fast.
    Avoids cutting mid-sentence for a more natural listening experience.
    """
    if len(text) <= max_chars:
        return text
    # Find the last sentence end within the limit
    truncated = text[:max_chars]
    last_end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
    if last_end > int(max_chars * 0.6):
        return truncated[:last_end + 1]
    return truncated.rstrip() + "…"


def generate_speech(text: str, voice: str = "alloy") -> bytes:
    """
    Generate MP3 audio bytes from text using gpt-4o-mini-tts.
    - Streams chunks from Navigate Labs for improved time-to-first-byte.
    - Truncates long texts to ~1200 chars to keep inference fast.
    - Supports English and Tamil without any extra configuration.
    """
    client = _get_client()
    clean_text = _truncate_for_speech(text)

    buffer = io.BytesIO()
    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=clean_text,
            response_format="mp3",
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4096):
                buffer.write(chunk)
    except Exception as e:
        raise RuntimeError(f"gpt-4o-mini-tts speech generation failed: {str(e)}")

    buffer.seek(0)
    return buffer.read()

