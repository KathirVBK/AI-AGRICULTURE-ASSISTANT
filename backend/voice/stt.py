"""
AgriSense-AI — stt.py
"""

import sounddevice as sd
from scipy.io.wavfile import write
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Client (Using OpenAI SDK for Groq consistency)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com")
)


def record_audio(filename="temp.wav", fs=16000):
    """Record audio until user presses Enter to stop."""
    print("\n🎤 Press ENTER to start recording...")
    input()

    print("🔴 Recording... Press ENTER to stop.")

    # Store audio chunks
    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            print(f"⚠️ {status}")
        audio_data.append(indata.copy())

    try:
        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
            input()  # Wait for Enter to stop

        print("✅ Recording stopped")

        if not audio_data:
            print("❌ No audio recorded")
            return ""

        # Concatenate all chunks
        audio_data = np.concatenate(audio_data, axis=0)

        # Save file
        write(filename, fs, audio_data)
        return filename

    except Exception as e:
        print(f"❌ Recording Error: {e}")
        return ""


def speech_to_text(audio_path):
    """Transcribe audio using Groq Whisper-3 (via OpenAI client)"""
    try:
        if not os.path.exists(audio_path):
            print(f"❌ Error: Audio file {audio_path} not found")
            return ""

        print("👂 Transcribing...")

        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, file),
                model="whisper-large-v3",
                response_format="text"
            )

        return transcription

    except Exception as e:
        print(f"❌ STT Error: {e}")
        return ""
