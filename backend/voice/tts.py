"""
AgriSense-AI — voice/tts.py
Text-to-Speech implementation via raw HTTP requests.
Returns audio bytes to be streamed back to the frontend.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def generate_speech(text: str) -> bytes:
    """
    Calls the TTS API and returns the raw audio bytes (MP3).
    The caller (FastAPI endpoint) is responsible for streaming
    these bytes back to the browser — no local file or playback.
    """
    url = os.getenv("TTS_BASE_URL")
    api_key = os.getenv("TTS_API_KEY")

    if not url or not api_key:
        raise ValueError("TTS_BASE_URL or TTS_API_KEY is not configured in environment variables.")

    # Ensure URL ends with the correct path
    endpoint = url.rstrip('/')
    if not endpoint.endswith('/v1/audio/speech'):
        endpoint += "/v1/audio/speech"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini-tts",
        "input": text,
        "voice": "alloy"
    }

    response = requests.post(endpoint, json=payload, headers=headers, timeout=20)

    if response.status_code != 200:
        raise RuntimeError(f"TTS API Error ({response.status_code}): {response.text}")

    return response.content  # Raw MP3 bytes
