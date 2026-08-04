import os
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth_module
import logging

logger = logging.getLogger(__name__)

# Path to the service account JSON file
SERVICE_ACCOUNT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'firebase_service_account.json'
)

import json

def initialize_firebase():
    """Initialize Firebase Admin SDK (idempotent)."""
    if not firebase_admin._apps:
        try:
            # 1. Try loading from individual environment variables first (reconstructing the certificate dict)
            project_id = os.getenv("FIREBASE_PROJECT_ID")
            private_key = os.getenv("FIREBASE_PRIVATE_KEY")
            client_email = os.getenv("FIREBASE_CLIENT_EMAIL")

            if project_id and private_key and client_email:
                logger.info("Initializing Firebase Admin SDK from individual environment variables...")
                # Reconstruct the private key replacing escaped newlines
                formatted_private_key = private_key.replace("\\n", "\n")
                service_account_info = {
                    "type": "service_account",
                    "project_id": project_id,
                    "private_key": formatted_private_key,
                    "client_email": client_email,
                    "token_uri": "https://oauth2.googleapis.com/token"
                }
                cred = credentials.Certificate(service_account_info)
            # 2. Try loading from single environment variable (fallback)
            elif os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"):
                logger.info("Initializing Firebase Admin SDK from FIREBASE_SERVICE_ACCOUNT_JSON...")
                service_account_info = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"))
                cred = credentials.Certificate(service_account_info)
            # 3. Fall back to local file (for local development)
            elif os.path.exists(SERVICE_ACCOUNT_PATH):
                logger.info("Initializing Firebase Admin SDK from local file...")
                cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            else:
                raise FileNotFoundError(
                    f"Firebase credentials not found. Please set individual environment variables "
                    f"(FIREBASE_PROJECT_ID, FIREBASE_PRIVATE_KEY, FIREBASE_CLIENT_EMAIL) "
                    f"or ensure the local file exists at: {SERVICE_ACCOUNT_PATH}"
                )
            
            firebase_admin.initialize_app(cred)
            logger.info("✔️ Firebase Admin SDK initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Firebase Admin SDK: {e}")
            raise

def verify_firebase_token(id_token: str) -> dict:
    """
    Verify a Firebase ID token and return the decoded payload.
    Returns a dict with: uid, email, name, picture
    Raises ValueError if the token is invalid.
    """
    try:
        decoded_token = firebase_auth_module.verify_id_token(id_token)
        return {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name", ""),
            "picture": decoded_token.get("picture", ""),
        }
    except firebase_auth_module.ExpiredIdTokenError:
        raise ValueError("Firebase token has expired.")
    except firebase_auth_module.InvalidIdTokenError:
        raise ValueError("Invalid Firebase token.")
    except Exception as e:
        raise ValueError(f"Token verification failed: {str(e)}")
