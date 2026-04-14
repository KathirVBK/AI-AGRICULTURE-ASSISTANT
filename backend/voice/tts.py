"""
AgriSense-AI — voice/tts.py
Text-to-Speech implementation via raw HTTP requests.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def speak(text):
    """
    Directly calls the TTS API via raw POST request.
    Bypasses OpenAI SDK to handle custom proxy/auth behaviors.
    """
    try:
        url = os.getenv("TTS_BASE_URL")
        api_key = os.getenv("TTS_API_KEY")

        if not url or not api_key:
            print("❌ TTS Error: Missing API_KEY or BASE_URL in .env")
            return

        # Ensure URL ends correctly for the proxy
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

        print(f"🔊 Calling TTS Proxy: {endpoint}...")
        # Add timeout to avoid hanging indefinitely
        response = requests.post(endpoint, json=payload, headers=headers, timeout=15)

        if response.status_code != 200:
            print(f"❌ TTS Error ({response.status_code}):", response.text)
            return

        file_path = "output.mp3"
        with open(file_path, "wb") as f:
            f.write(response.content)

        print("🔊 Playing response...")
        # Play audio (Windows)
        os.startfile(file_path)

    except Exception as e:
        print(f"❌ TTS Exception: {e}")
