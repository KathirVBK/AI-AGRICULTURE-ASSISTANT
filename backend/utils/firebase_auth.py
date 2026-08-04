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
            # 1. Try loading from environment variable first (for production)
            env_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
            if env_json:
                logger.info("Initializing Firebase Admin SDK from environment variable...")
                service_account_info = json.loads(env_json)
                cred = credentials.Certificate(service_account_info)
            # 2. Fall back to local file (for local development)
            elif os.path.exists(SERVICE_ACCOUNT_PATH):
                logger.info("Initializing Firebase Admin SDK from local file...")
                cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            else:
                raise FileNotFoundError(
                    f"Firebase credentials not found. Please set the "
                    f"FIREBASE_SERVICE_ACCOUNT_JSON environment variable or ensure "
                    f"the file exists at: {SERVICE_ACCOUNT_PATH}"
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
