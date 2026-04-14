from fastapi import FastAPI, HTTPException, APIRouter, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
import logging
from google.oauth2 import id_token
from google.auth.transport import requests
import os
from groq import Groq

from core.pipeline import run_query
from core.database import get_db, User
from utils.auth import get_password_hash, verify_password, create_access_token, decode_access_token
from rag.vector_store import warmup

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AgriSense AI API")

@app.on_event("startup")
async def startup_event():
    """Trigger pre-warming of AI models during server startup."""
    logger.info("⚡ [BOOT] Starting AgriSense AI Pipeline pre-warmup...")
    try:
        warmup()
        logger.info("🚀 [BOOT] AgriSense AI Pipeline is fully optimized and ready.")
    except Exception as e:
        logger.error(f"❌ [BOOT] Pre-warmup failed: {str(e)}")

# Initialize Groq client for STT
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))



oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# 🔗 Google Config
GOOGLE_CLIENT_ID = "127247653405-9hfelc3b0r3lelnot14i4t6dqtna8thh.apps.googleusercontent.com"

# 🧠 Simple In-Memory Session Store
session_store: Dict[str, List[dict]] = {}

api_router = APIRouter(prefix="/api")

# ── Pydantic Models ──────────────────────────────────────────
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class GoogleLoginRequest(BaseModel):
    token: str

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

# ── Auth Logic ──────────────────────────────────────────────
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    email: str = payload.get("sub")
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ── Routes ──────────────────────────────────────────────────
@api_router.get("/")
async def api_root():
    return {"message": "AgriSense API is active"}

@api_router.post("/auth/signup")
async def signup(request: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == request.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(request.password)
    new_user = User(
        email=request.email,
        hashed_password=hashed_password,
        full_name=request.full_name
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = create_access_token(data={"sub": new_user.email})
    return {"access_token": access_token, "token_type": "bearer", "user": {"email": new_user.email, "full_name": new_user.full_name}}

@api_router.post("/auth/login")
async def login(request: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer", "user": {"email": user.email, "full_name": user.full_name}}

@api_router.post("/auth/google")
async def google_login(request: GoogleLoginRequest, db: Session = Depends(get_db)):
    try:
        # ✅ Verify the Google Token
        idinfo = id_token.verify_oauth2_token(
            request.token, 
            requests.Request(), 
            GOOGLE_CLIENT_ID
        )

        # Extract user info
        email = idinfo['email']
        full_name = idinfo.get('name', '')
        avatar_url = idinfo.get('picture', '')
        
        logger.info(f"Successfully verified Google user: {email}")

        # Check if user exists in our DB
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Create a new user if they don't exist
            user = User(
                email=email, 
                full_name=full_name,
                avatar_url=avatar_url,
                hashed_password="GOOGLE_AUTH_USER" # Placeholder for Google users
            )
            db.add(user)
        else:
            # Update existing user's avatar if it changed
            user.avatar_url = avatar_url
            
        db.commit()
        db.refresh(user)
        
        # Issue our own JWT access token
        access_token = create_access_token(data={"sub": user.email})
        return {
            "access_token": access_token, 
            "token_type": "bearer", 
            "user": {
                "email": user.email, 
                "full_name": user.full_name,
                "avatar_url": user.avatar_url
            }
        }
        
    except ValueError as e:
        # Invalid token
        logger.error(f"Google login failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid Google token")
    except Exception as e:
        logger.error(f"Unexpected error in Google login: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during login")

@api_router.get("/auth/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {"email": current_user.email, "full_name": current_user.full_name}

@api_router.post("/chat")
async def chat(request: ChatRequest):
    # Optional: allow both auth and guest if needed, 
    # but based on the request, we should probably protect this.
    try:
        session_id = request.session_id
        history = session_store.get(session_id, [])
        
        result = run_query(
            query=request.query,
            history=history,
            include_trace=True
        )
        
        if not isinstance(result, dict):
            answer = result
            follow_ups = []
            trace = {}
        else:
            answer = result.get("response", "")
            follow_ups = result.get("follow_ups", [])
            trace = result.get("trace", {})

        history.append({"role": "user", "content": request.query})
        history.append({"role": "assistant", "content": answer})
        if len(history) > 20:
            history = history[-20:]
        
        session_store[session_id] = history
        
        return {
            "answer": answer,
            "follow_ups": follow_ups,
            "trace": trace,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in session_store:
        del session_store[session_id]
        return {"message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")

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
