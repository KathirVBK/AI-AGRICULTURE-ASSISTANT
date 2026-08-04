from fastapi import FastAPI, HTTPException, APIRouter, Depends, status, UploadFile, File
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
import logging
import os
import json
import datetime
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils.firebase_auth import initialize_firebase, verify_firebase_token

# Initialize Firebase Admin SDK at startup
initialize_firebase()

from core.pipeline import run_query
from core.database import get_db, User, ChatSession, ChatMessage
from rag.vector_store import warmup

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Trigger pre-warming of AI models during server startup."""
    logger.info("⚡ [BOOT] Starting AgriSense AI Pipeline pre-warmup...")
    try:
        warmup()
        logger.info("🚀 [BOOT] AgriSense AI Pipeline is fully optimized and ready.")
    except Exception as e:
        logger.error(f"❌ [BOOT] Pre-warmup failed: {str(e)}")
    yield
 
app = FastAPI(title="AgriSense AI API", lifespan=lifespan)

# Initialize Groq client for STT
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))



# 🔐 Firebase Bearer token extractor
bearer_scheme = HTTPBearer(auto_error=False)

# 🧠 In-Memory Stores
session_store: Dict[str, List[dict]] = {}
soil_report_store: Dict[str, dict] = {}

# Ensure upload directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "soil_reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)

api_router = APIRouter(prefix="/api")

from utils.validator import extract_entities
import pypdf

# ── Pydantic Models ──────────────────────────────────────────
class FirebaseLoginRequest(BaseModel):
    id_token: str

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

class TTSRequest(BaseModel):
    text: str

class CreateSessionRequest(BaseModel):
    session_id: str
    title: str = "New Consultation"

class UpdateSessionRequest(BaseModel):
    title: str

# ── Auth Logic ──────────────────────────────────────────────
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        token_data = verify_firebase_token(credentials.credentials)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.email == token_data["email"]).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found. Please login via /api/auth/firebase first.")
    return user

# ── Routes ──────────────────────────────────────────────────
@api_router.get("/")
async def api_root():
    return {"message": "AgriSense API is active"}

@api_router.post("/auth/firebase")
async def firebase_login(request: FirebaseLoginRequest, db: Session = Depends(get_db)):
    """
    Verify a Firebase ID token and upsert the user in the local DB.
    Works for both Email/Password and Google sign-in methods.
    """
    try:
        token_data = verify_firebase_token(request.id_token)
    except ValueError as e:
        logger.error(f"Firebase token verification failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    email = token_data["email"]
    full_name = token_data.get("name", "")
    avatar_url = token_data.get("picture", "")
    firebase_uid = token_data["uid"]

    try:
        # Upsert user in local DB
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(
                email=email,
                full_name=full_name,
                avatar_url=avatar_url,
                google_id=firebase_uid,
                hashed_password="FIREBASE_AUTH_USER"
            )
            db.add(user)
            logger.info(f"✅ New Firebase user created: {email}")
        else:
            # Update profile info if changed
            user.avatar_url = avatar_url or user.avatar_url
            user.full_name = full_name or user.full_name
            user.google_id = firebase_uid
            logger.info(f"✅ Existing Firebase user logged in: {email}")

        db.commit()
        db.refresh(user)

        return {
            "user": {
                "email": user.email,
                "full_name": user.full_name,
                "avatar_url": user.avatar_url
            }
        }
    except Exception as e:
        logger.error(f"Database error during Firebase login: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during login")

@api_router.get("/auth/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "full_name": current_user.full_name,
        "avatar_url": current_user.avatar_url
    }

@api_router.post("/soil-report/upload")
async def upload_soil_report(file: UploadFile = File(...), session_id: str = "default"):
    """
    Upload and parse Soil Test document (PDF or Image).
    Extracts report text, entities, and stores context for the session.
    """
    try:
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in [".pdf", ".png", ".jpg", ".jpeg", ".webp"]:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PDF or Image (.pdf, .png, .jpg, .jpeg).")

        file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{filename}")
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        extracted_text = ""

        if file_ext == ".pdf":
            # Extract text from PDF
            reader = pypdf.PdfReader(file_path)
            pdf_texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t: pdf_texts.append(t)
            extracted_text = "\n".join(pdf_texts).strip()
        else:
            # For images, extract text using Pillow / OCR fallback
            try:
                from PIL import Image
                img = Image.open(file_path)
                # Store basic metadata description if OCR is not active
                extracted_text = f"Soil Test Image Document: {filename} (Image dimensions: {img.width}x{img.height} pixels)."
            except Exception as ie:
                extracted_text = f"Soil Test Image Document: {filename}."

        if not extracted_text:
            extracted_text = f"Uploaded Soil Test Document: {filename}"

        entities = extract_entities(extracted_text)

        report_data = {
            "filename": filename,
            "file_path": file_path,
            "text": extracted_text,
            "entities": entities,
            "session_id": session_id
        }

        soil_report_store[session_id] = report_data

        logger.info(f"✅ Successfully uploaded & parsed soil report for session '{session_id}': {filename}")

        return {
            "status": "success",
            "message": f"Successfully uploaded and analyzed soil report: {filename}",
            "report": {
                "filename": filename,
                "entities": entities,
                "summary": extracted_text[:400]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading soil report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process soil report document: {str(e)}")

@api_router.get("/soil-report/{session_id}")
async def get_session_soil_report(session_id: str):
    report = soil_report_store.get(session_id)
    if not report:
        return {"report": None}
    return {
        "report": {
            "filename": report.get("filename"),
            "entities": report.get("entities"),
            "summary": report.get("text", "")[:400]
        }
    }

@api_router.delete("/soil-report/{session_id}")
async def delete_session_soil_report(session_id: str):
    if session_id in soil_report_store:
        del soil_report_store[session_id]
        return {"message": "Soil report document removed."}
    return {"message": "No active soil report found."}

@api_router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        session_id = request.session_id

        # ── Hydrate in-memory context from DB if not already loaded ──
        if session_id not in session_store:
            db_session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == current_user.id
            ).first()
            if db_session:
                session_store[session_id] = [
                    {"role": msg.role, "content": msg.content}
                    for msg in db_session.messages
                ][-20:]  # keep last 20 for context window

        history = session_store.get(session_id, [])

        # Retrieve uploaded soil report for this session if available
        soil_report = soil_report_store.get(session_id)
        soil_report_context = soil_report.get("text") if soil_report else None

        result = run_query(
            query=request.query,
            history=history,
            include_trace=True,
            soil_report_context=soil_report_context
        )

        if not isinstance(result, dict):
            answer = result
            follow_ups = []
            trace = {}
        else:
            answer = result.get("response", "")
            follow_ups = result.get("follow_ups", [])
            trace = result.get("trace", {})

        # ── Update in-memory context ──
        history.append({"role": "user", "content": request.query})
        history.append({"role": "assistant", "content": answer})
        if len(history) > 20:
            history = history[-20:]
        session_store[session_id] = history

        # ── Persist messages to DB ──
        now_ts = datetime.datetime.now().strftime("%I:%M %p")
        db_session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        ).first()
        
        if not db_session:
            # Auto-create the session if it was not created/persisted yet
            short_title = request.query[:28] + "…" if len(request.query) > 28 else request.query
            db_session = ChatSession(
                id=session_id,
                user_id=current_user.id,
                title=short_title
            )
            db.add(db_session)
            db.commit()
            db.refresh(db_session)

        db.add(ChatMessage(
            session_id=session_id,
            role="user",
            content=request.query,
            timestamp=now_ts
        ))
        db.add(ChatMessage(
            session_id=session_id,
            role="assistant",
            content=answer,
            follow_ups=json.dumps(follow_ups) if follow_ups else None,
            trace=json.dumps(trace) if trace else None,
            timestamp=now_ts
        ))
        db_session.updated_at = datetime.datetime.utcnow()
        db.commit()

        return {
            "answer": answer,
            "follow_ups": follow_ups,
            "trace": trace,
            "status": "success",
            "soil_report_attached": bool(soil_report)
        }
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Session CRUD Endpoints ──────────────────────────────────

@api_router.get("/sessions")
async def get_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all sessions with messages for the current user."""
    db_sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    result = []
    for s in db_sessions:
        result.append({
            "id": s.id,
            "title": s.title,
            "timestamp": s.updated_at.isoformat(),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "followUps": json.loads(msg.follow_ups) if msg.follow_ups else [],
                    "trace": json.loads(msg.trace) if msg.trace else None,
                    "timestamp": msg.timestamp
                }
                for msg in s.messages
            ]
        })
    return {"sessions": result}

@api_router.post("/sessions")
async def create_session(
    request: CreateSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new chat session for the current user."""
    # Avoid duplicate if frontend retries
    existing = db.query(ChatSession).filter(ChatSession.id == request.session_id).first()
    if existing:
        return {"session": {"id": existing.id, "title": existing.title, "timestamp": existing.updated_at.isoformat(), "messages": []}}

    new_session = ChatSession(
        id=request.session_id,
        user_id=current_user.id,
        title=request.title
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {
        "session": {
            "id": new_session.id,
            "title": new_session.title,
            "timestamp": new_session.created_at.isoformat(),
            "messages": []
        }
    }

@api_router.put("/sessions/{session_id}")
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update the title of a session."""
    s = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    s.title = request.title
    s.updated_at = datetime.datetime.utcnow()
    db.commit()
    return {"message": "Session updated"}

@api_router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Permanently delete a session and all its messages."""
    s = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if s:
        db.delete(s)
        db.commit()
    session_store.pop(session_id, None)
    soil_report_store.pop(session_id, None)
    return {"message": f"Session {session_id} deleted"}

@api_router.delete("/sessions/{session_id}/messages")
async def clear_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clear all messages in a session and reset its title."""
    s = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if s:
        for msg in list(s.messages):
            db.delete(msg)
        s.title = "New Consultation"
        s.updated_at = datetime.datetime.utcnow()
        db.commit()
    session_store.pop(session_id, None)
    soil_report_store.pop(session_id, None)
    return {"message": f"Session {session_id} messages cleared"}

@api_router.delete("/session/{session_id}")
async def clear_session_legacy(session_id: str):
    """Legacy endpoint — kept for backward compatibility."""
    session_store.pop(session_id, None)
    soil_report_store.pop(session_id, None)
    return {"message": f"Session {session_id} cleared"}

@api_router.post("/stt")
async def speech_to_text(audio_file: UploadFile = File(...)):
    if not groq_client.api_key:
        raise HTTPException(status_code=500, detail="Groq API key not configured for STT.")

    try:
        # Save the uploaded audio file temporarily
        temp_audio_path = f"temp_{audio_file.filename}"
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        # Transcribe using Groq's Whisper model with refined parameters
        with open(temp_audio_path, "rb") as audio:
            transcript = groq_client.audio.transcriptions.create(
                file=(audio_file.filename, audio.read(), audio_file.content_type),
                model="whisper-large-v3",
                # Enhanced prompt with more context and common crop names
                prompt=(
                    "The user is asking about agriculture, crops (like rice, wheat, maize, sugarcane, cotton), "
                    "soil health (pH, NPK, Nitrogen, Phosphorus, Potassium), pests, fertilizers, and farming practices. "
                    "Please transcribe exactly what is said with correct punctuation."
                ),
                language="en", # Force English for better accuracy if appropriate
                temperature=0.0, # Most accurate/deterministic output
                response_format="json"
            )
        
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        return {"text": transcript.text}
    except Exception as e:
        logger.error(f"Error during speech-to-text transcription: {str(e)}")
        # Attempt cleanup on failure
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")

@api_router.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        from voice.tts import generate_speech
        audio_bytes = generate_speech(request.text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"Error during text-to-speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Enable CORS for React frontend
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    # Render provides the PORT environment variable
    port = int(os.getenv("PORT", 8000))
    # Use 0.0.0.0 to bind to all available interfaces in a container/server environment
    uvicorn.run(app, host="0.0.0.0", port=port)
