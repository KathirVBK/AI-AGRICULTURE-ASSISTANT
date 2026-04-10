<div align="center">
  
# 🌱 AgriSense Precision AI

**Intelligent Agricultural Advisory & Management Platform**

[![React](https://img.shields.io/badge/Frontend-React.js-61DAFB?logo=react&logoColor=black)](#)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](#)
[![Llama-3](https://img.shields.io/badge/LLM-Llama_3-0466c8?logo=meta&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Language-Python_3.10+-3776AB?logo=python&logoColor=white)](#)

*AgriSense AI provides real-time, context-aware, scientific agricultural advice using a powerful mixture of Global Web Search and Local RAG (Retrieval-Augmented Generation) combined with ultra-fast LLM inference.*

</div>

## ✨ Features
- **Strict Agricultural Expert Mode**: Refuses non-agricultural topics and impossible scenarios using a strict, multi-pass reasoning pipeline.
- **RAG & Global Web Search**: Synthesizes verified data directly from local datasets and live global trends.
- **Voice Capabilities**: Record voice queries and get intelligent synthesized Voice/Text output back.
- **Beautiful UI**: Modern glassmorphism design with a fully responsive mobile layout, Light/Dark Modes, and localized user authentication via Google OAuth.
- **Strict Output Parsing**: Beautifully parsed formatting for Expert Analysis, Recommendations, Follow-up Questions, and Expert Cautions.

---

## 🛠️ Architecture 
AgriSense utilizes a dual-stack configuration:
1. **Backend (`/` root)** - Powered by **FastAPI**. Handles LLM (Groq) connections, vector similarity searching (RAG pipeline), duckduckgo live search (`ddgs`), authentication token minting, and the SQLite chat persistence database.
2. **Frontend (`/frontend`)** - Powered by **React**. Provides the rich dashboard UI, Google OAuth layer, voice-recording interface, markdown rendering, and local session management.

---

## 🚀 Step-by-Step Setup Guide

### Prerequisites
Before you start, ensure you have the following installed on your machine:
* **Python 3.10+** (For the backend)
* **Node.js (v16+) and npm** (For the frontend)
* **Git** 

### 1. Clone the Repository
```bash
git clone https://github.com/KathirVBK/AI-AGRICULTURE-ASSISTANT.git
cd AI-AGRICULTURE-ASSISTANT
```

### 2. Set Up the Backend (Python FastAPI)

1. **Create a Virtual Environment**:
   It is highly recommended you keep dependencies isolated.
   ```bash
   python -m venv .venv
   ```
   * **Windows**: `.\.venv\Scripts\activate`
   * **Mac/Linux**: `source .venv/bin/activate`

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Set Up API Keys (`.env`)

The core intelligence of AgriSense runs entirely on Groq inference and a standalone TTS (Text-to-Speech) service.

1. Create a file named `.env` in the root folder (`AI-AGRICULTURE-ASSISTANT/`).
2. Add the following structure to the file:

```env
# Groq API Key for LLM Generation (Llama-3 models)
GROQ_API_KEY=your_groq_api_key_here

# Text To Speech API (Navigate Labs or alternative compatible)
TTS_API_KEY=your_tts_api_key_here
TTS_BASE_URL=https://apidev.navigatelabsai.com/
```

#### How to get the `GROQ_API_KEY`:
1. Go to the [GroqCloud Developer Console](https://console.groq.com/).
2. Create an account or log in.
3. Once in the dashboard, navigate to **API Keys** on the left menu.
4. Click **Create API Key**.
5. Copy the generated key (it usually starts with `gsk_`) and paste it into `.env` under `GROQ_API_KEY=...`.

#### How to setup Google OAuth Client ID (Frontend):
*If you are running locally, the default client ID in the React app should work for `localhost:3000`. If you deploy publicly, you will need to replace `GOOGLE_CLIENT_ID` in `frontend/src/App.js` with your own from the [Google Cloud Console](https://console.cloud.google.com/).*

### 4. Start the Backend Server

Ensure you are still in your virtual environment (`.venv` activated).

```bash
# Starts the FastAPI server on port 8000
python app.py
```
*The backend should now be running at `http://127.0.0.1:8000`.*

### 5. Set Up the Frontend (React application)

In a **new terminal window**, navigate to the frontend folder, install dependencies, and start the app.

```bash
cd frontend

# Install Node dependencies
npm install

# Start the React development server
npm start
```
*The app should automatically launch in your default web browser at `http://localhost:3000`.*

---

## 💡 How to Use the App
1. **Login**: Use the Google sign-in wrapper to authenticate, or click **"Continue with Demo User"** for a quick preview.
2. **Consulting**: Type queries like *"How to manage soil with pH 5.5 in Tamil Nadu?"* or use the Quick Queries grid.
3. **Voice Input**: Click the microphone icon to natively record your voice instead of typing.
4. **Theme Toggle**: Switch between Light Mode and Dark Mode using the Sun/Moon icon in the top right.
5. **Manage Sessions**: Check out, search, and delete your older consultations from the collapsible sidebar. 

## 🔒 Security
- All keys inside `.env` are protected via the provided `.gitignore`.
- **NEVER** commit your `.env` file to GitHub.

---
<div align="center">
<i>Built to grow a smarter, more sustainable world.</i>
</div>
