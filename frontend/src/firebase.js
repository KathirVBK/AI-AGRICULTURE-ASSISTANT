import { initializeApp } from 'firebase/app';
import {
  getAuth,
  GoogleAuthProvider,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  indexedDBLocalPersistence,
  inMemoryPersistence,
  setPersistence,
} from 'firebase/auth';

const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID,
  measurementId: process.env.REACT_APP_FIREBASE_MEASUREMENT_ID,
};

// Initialize Firebase app
const app = initializeApp(firebaseConfig);

// Initialize Firebase Auth
export const auth = getAuth(app);

/**
 * Clears all Firebase-related IndexedDB databases.
 * This resolves the "Database is closing/hidden" error caused by
 * corrupted IndexedDB state left from abrupt browser closures.
 */
async function clearFirebaseIndexedDB() {
  if (!window.indexedDB) return;
  const dbNames = [
    'firebaseLocalStorageDb',
    'firebase-heartbeat-database',
    'firebase-installations-database',
  ];
  for (const name of dbNames) {
    try {
      await new Promise((resolve, reject) => {
        const req = window.indexedDB.deleteDatabase(name);
        req.onsuccess = resolve;
        req.onerror = reject;
        req.onblocked = resolve; // resolve anyway if blocked
      });
    } catch (_) {
      // Ignore individual failures — best effort cleanup
    }
  }
}

/**
 * Configures Auth persistence.
 * Falls back to inMemoryPersistence if IndexedDB is broken.
 */
async function setupAuthPersistence() {
  try {
    await setPersistence(auth, indexedDBLocalPersistence);
  } catch (err) {
    const msg = err?.message || '';
    if (
      msg.includes('closing') ||
      msg.includes('hidden') ||
      msg.includes('IndexedDB')
    ) {
      console.warn('[Firebase] IndexedDB broken — clearing and retrying...', err);
      await clearFirebaseIndexedDB();
      try {
        // Retry with IndexedDB after clearing
        await setPersistence(auth, indexedDBLocalPersistence);
        console.info('[Firebase] IndexedDB recovered. Persistence restored.');
      } catch (retryErr) {
        // Final fallback: use in-memory persistence (session only)
        console.warn('[Firebase] Falling back to in-memory persistence.', retryErr);
        await setPersistence(auth, inMemoryPersistence);
      }
    }
  }
}

// Run persistence setup on module load (non-blocking)
setupAuthPersistence();

// Google provider
export const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: 'select_account' });

// Auth helpers
export {
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  GoogleAuthProvider,
};

export default app;
