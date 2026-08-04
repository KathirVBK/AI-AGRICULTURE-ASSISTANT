import React, { useState } from 'react';
import { Mail, Lock, User, ArrowRight, Sprout, Leaf } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import {
  auth,
  googleProvider,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
} from '../../firebase';
import './AuthPage.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api';

const AuthPage = ({ onLoginSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({ email: '', password: '', full_name: '' });
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError('');
  };

  /**
   * After Firebase signs in the user, call our backend to sync user profile.
   */
  const syncWithBackend = async (firebaseUser) => {
    const idToken = await firebaseUser.getIdToken();
    const response = await axios.post(`${API_BASE_URL}/auth/firebase`, { id_token: idToken });
    const { user } = response.data;
    localStorage.setItem('agri_user', JSON.stringify(user));
    onLoginSuccess(user);
  };

  /** Maps raw Firebase/network errors to user-friendly messages */
  const getFriendlyError = (err) => {
    const raw = err?.response?.data?.detail || err?.message || '';
    if (raw.includes('closing') || raw.includes('hidden') || raw.includes('IndexedDB')) {
      return 'Session storage error. Please refresh the page and try again.';
    }
    const codeMap = {
      'auth/user-not-found': 'No account found with this email.',
      'auth/wrong-password': 'Incorrect password. Please try again.',
      'auth/email-already-in-use': 'An account with this email already exists.',
      'auth/invalid-email': 'Please enter a valid email address.',
      'auth/weak-password': 'Password must be at least 6 characters.',
      'auth/too-many-requests': 'Too many attempts. Please try again later.',
      'auth/network-request-failed': 'Network error. Check your connection.',
      'auth/invalid-credential': 'Invalid email or password.',
    };
    return codeMap[err?.code] || raw || 'Authentication failed.';
  };

  /** Email / Password sign-in or sign-up */
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    try {
      let userCredential;
      if (isLogin) {
        userCredential = await signInWithEmailAndPassword(auth, formData.email, formData.password);
      } else {
        userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
      }
      await syncWithBackend(userCredential.user);
    } catch (err) {
      setError(getFriendlyError(err));
    } finally {
      setIsLoading(false);
    }
  };

  /** Google Sign-In via Firebase popup */
  const handleGoogleSignIn = async () => {
    setIsLoading(true);
    setError('');
    try {
      const result = await signInWithPopup(auth, googleProvider);
      await syncWithBackend(result.user);
    } catch (err) {
      if (err.code !== 'auth/popup-closed-by-user') {
        setError(getFriendlyError(err));
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-glass">
        <div className="auth-side-brand">
          <div className="auth-brand-logo">
            <Sprout size={48} color="white" />
          </div>
          <h1>AgriSense AI</h1>
          <p>Scientific Precision for Modern Agriculture</p>
          <div className="auth-features">
            <div className="feature-item"><Leaf size={16} /> <span>Smart Crop Analysis</span></div>
            <div className="feature-item"><Leaf size={16} /> <span>Global Knowledge Base</span></div>
            <div className="feature-item"><Leaf size={16} /> <span>Real-time Insights</span></div>
          </div>
        </div>

        <div className="auth-form-side">
          <AnimatePresence mode="wait">
            <motion.div
              key={isLogin ? 'login' : 'signup'}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="form-content"
            >
              <h2>{isLogin ? 'Welcome Back' : 'Join AgriSense'}</h2>
              <p className="subtitle">
                {isLogin
                  ? 'Access your personalized agricultural guidance'
                  : 'Start your journey towards high-yield precision farming'}
              </p>

              <form onSubmit={handleSubmit}>
                {!isLogin && (
                  <div className="input-group">
                    <User className="input-icon" size={18} />
                    <input
                      type="text"
                      name="full_name"
                      placeholder="Full Name"
                      value={formData.full_name}
                      onChange={handleChange}
                    />
                  </div>
                )}
                <div className="input-group">
                  <Mail className="input-icon" size={18} />
                  <input
                    type="email"
                    name="email"
                    placeholder="Email Address"
                    value={formData.email}
                    onChange={handleChange}
                    required
                  />
                </div>
                <div className="input-group">
                  <Lock className="input-icon" size={18} />
                  <input
                    type="password"
                    name="password"
                    placeholder="Password"
                    value={formData.password}
                    onChange={handleChange}
                    required
                  />
                </div>

                {error && <div className="auth-error">{error}</div>}

                <button type="submit" className="auth-submit-btn" disabled={isLoading}>
                  {isLoading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
                  <ArrowRight size={18} />
                </button>
              </form>

              <div className="auth-divider"><span>OR</span></div>

              <div className="google-auth-wrapper">
                <button
                  className="google-demo-btn"
                  onClick={handleGoogleSignIn}
                  disabled={isLoading}
                  type="button"
                >
                  <img
                    src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg"
                    alt="Google"
                    width={20}
                    height={20}
                  />
                  {isLoading ? 'Signing in...' : 'Continue with Google'}
                </button>
              </div>

              <p className="toggle-auth">
                {isLogin ? "Don't have an account?" : 'Already have an account?'}{' '}
                <button onClick={() => setIsLogin(!isLogin)} className="toggle-link">
                  {isLogin ? 'Sign Up' : 'Sign In'}
                </button>
              </p>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;
