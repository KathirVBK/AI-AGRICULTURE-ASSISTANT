import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import {
  Send,
  Sprout,
  Trash2,
  Plus,
  MessageSquare,
  X,
  ChevronRight,
  Database,
  Search,
  CheckCircle2,
  Leaf,
  LogOut,
  User as UserIcon,
  Mic,
  StopCircle,
  Settings,
  Sun,
  Moon,
  Copy,
  Check,
  Menu,
  AlertTriangle,
  Paperclip,
  FileText,
  UploadCloud,
  Volume2,
  Square,
} from 'lucide-react';
import './App.css';
import AuthPage from './components/Auth/AuthPage';
import { auth, onAuthStateChanged, signOut as firebaseSignOut } from './firebase';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api';

const QUICK_QUERIES = [
  "What crops are best for summer in Punjab?",
  "How to manage soil with pH 5.5?",
  "Organic control for Rice Stem Borer",
  "Optimal NPK for Sugarcane in Tamil Nadu",
  "High-yield tomato varieties for greenhouse",
  "Best time to plant Arabica coffee",
  "Drought-resistant crops for Rajasthan",
  "Natural fertilizers for organic farming",
  "Symptoms of Nitrogen deficiency in Maize",
  "Integrated pest management for Cotton",
  "Hydroponic lettuce nutrient solution",
  "Winter wheat sowing depth in USA"
];

// Configure axios to always send a fresh Firebase ID token
axios.interceptors.request.use(
  async (config) => {
    const currentUser = auth.currentUser;
    if (currentUser) {
      const token = await currentUser.getIdToken();
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

/* ─── useTheme hook ─── */
function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem('agri_theme') || 'light');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('agri_theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  }, []);

  return { theme, setTheme, toggleTheme };
}

/* ─── Settings Modal ─── */
function SettingsModal({ isOpen, onClose, theme, setTheme, onClearAllSessions }) {
  if (!isOpen) return null;

  const handleClearAll = () => {
    if (window.confirm('Delete ALL chat sessions? This cannot be undone.')) {
      onClearAllSessions();
      onClose();
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={e => e.stopPropagation()}>
        <div className="settings-header">
          <h3>⚙ Settings</h3>
          <button className="settings-close" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="settings-body">
          {/* Theme Section */}
          <div>
            <div className="settings-section-label">Appearance</div>
            <div className="theme-options">
              <button
                className={`theme-option ${theme === 'light' ? 'active' : ''}`}
                onClick={() => setTheme('light')}
              >
                <div className="theme-preview theme-preview-light" />
                <Sun size={15} />
                Light Mode
              </button>
              <button
                className={`theme-option ${theme === 'dark' ? 'active' : ''}`}
                onClick={() => setTheme('dark')}
              >
                <div className="theme-preview theme-preview-dark" />
                <Moon size={15} />
                Dark Mode
              </button>
            </div>
          </div>

          {/* Danger Zone */}
          <div>
            <div className="settings-section-label">Danger Zone</div>
            <button className="settings-danger-btn" onClick={handleClearAll}>
              <AlertTriangle size={15} />
              Clear All Chat Sessions
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── CopyButton ─── */
function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (_) { }
  };

  return (
    <button className={`copy-btn ${copied ? 'copied' : ''}`} onClick={handleCopy} title="Copy response">
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

/* ─── AudioButton ─── */
function AudioButton({ text }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const audioRef = useRef(null);

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
    }
    setIsPlaying(false);
  };

  const handlePlayToggle = async (e) => {
    e.stopPropagation();
    if (isLoading) return;

    // Stop if currently playing
    if (isPlaying) {
      handleStop();
      return;
    }

    try {
      setIsLoading(true);
      const resp = await axios.post(`${API_BASE_URL}/tts`, { text }, {
        responseType: 'blob'
      });
      const audioUrl = URL.createObjectURL(resp.data);
      const audio = new Audio(audioUrl);
      audio.onended = () => { setIsPlaying(false); audioRef.current = null; };
      audio.onerror = () => { setIsPlaying(false); audioRef.current = null; };
      audioRef.current = audio;
      await audio.play();
      setIsPlaying(true);
    } catch (err) {
      console.error('TTS Failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const btnClass = `audio-btn${isPlaying ? ' playing' : ''}${isLoading ? ' loading' : ''}`;
  const title = isLoading ? 'Generating audio…' : isPlaying ? 'Stop' : 'Listen';

  return (
    <button
      className={btnClass}
      onClick={handlePlayToggle}
      title={title}
      disabled={isLoading}
      aria-label={title}
    >
      {isLoading ? (
        <span className="audio-spinner" />
      ) : isPlaying ? (
        <span className="sound-wave" aria-hidden="true">
          <span /><span /><span /><span /><span />
        </span>
      ) : (
        <Volume2 size={13} />
      )}
    </button>
  );
}

/* ─── Trace Block ─── */
function TraceBlock({ trace }) {
  const [isOpen, setIsOpen] = useState(false);
  if (!trace) return null;
  return (
    <div className="message-trace">
      <button className="trace-toggle" onClick={() => setIsOpen(!isOpen)}>
        <Database size={11} />
        {isOpen ? 'Hide Technical Trace' : 'View Technical Trace'}
      </button>
      {isOpen && (
        <div className="trace-content-mini">
          {trace.context?.entities?.location && <span className="tag">📍 {trace.context.entities.location}</span>}
          {trace.validated_crops?.length > 0 && (
            <span className="tag">🌾 {trace.validated_crops.slice(0, 3).join(', ')}</span>
          )}
          <span className="tag">🌐 Web: {trace.context?.web ? 'Yes' : 'No'}</span>
          <span className="tag">📖 RAG: {trace.context?.rag ? 'Yes' : 'No'}</span>
        </div>
      )}
    </div>
  );
}

/* ─── Typing Indicator ─── */
function TypingIndicator({ status }) {
  return (
    <div className="msg-row assistant">
      <div className="msg-with-avatar">
        <div className="avatar ai-avatar">
          <Leaf size={14} />
        </div>
        <div className="typing-bubble">
          <div className="dots">
            <span /><span /><span />
          </div>
          <span className="typing-label">{status || 'Thinking…'}</span>
        </div>
      </div>
    </div>
  );
}

/* ─── cleanAgriResponse ────────────────────────────────────
   Cleans up the raw LLM output on the frontend by removing
   asterisks and emojis, turning headers into clean text.
──────────────────────────────────────────────────────────── */
function cleanAgriResponse(text) {
  if (!text) return '';
  
  let cleaned = text;

  // Remove warning emojis frequently attached to cautions
  cleaned = cleaned.replace(/[\u26A0\u2757\u2755]/gu, '');

  // Target these known section headers, with or without asterisks
  const knownHeaders = [
    'Expert Analysis',
    'Primary Recommendations',
    'Supporting Details',
    'Follow-up Questions',
    'Expert Caution',
  ];
  
  // This regex matches things like "**Expert Analysis**" or "Expert Caution**"
  const headerRegex = new RegExp(`(?:\\*+)?(${knownHeaders.join('|')})(?:\\*+)?`, 'g');
  
  // Replace them with "Header: " so they look clean and standard
  cleaned = cleaned.replace(headerRegex, '$1:');
  
  // Remove any double spaces created by emoji removal
  cleaned = cleaned.replace(/ :\s+/g, ': ').trim();
  
  return cleaned;
}

/* ─── AppContent ─── */
function AppContent() {
  const { theme, setTheme, toggleTheme } = useTheme();

  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);

  // Listen to Firebase auth state changes for persistent sessions
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        const savedUser = localStorage.getItem('agri_user');
        if (savedUser) {
          setUser(JSON.parse(savedUser));
        } else {
          setUser({ email: firebaseUser.email, full_name: firebaseUser.displayName || '' });
        }
      } else {
        setUser(null);
        localStorage.removeItem('agri_user');
      }
      setAuthLoading(false);
    });
    return () => unsubscribe();
  }, []);

  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [query, setQuery] = useState('');
  const [randomQueries, setRandomQueries] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false); // mobile
  const [sessionSearch, setSessionSearch] = useState('');
  const [soilReport, setSoilReport] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const [sessionsLoaded, setSessionsLoaded] = useState(false);

  const createNewSession = useCallback(async () => {
    const newSession = {
      id: Date.now().toString(),
      title: 'New Consultation',
      messages: [],
      timestamp: new Date().toISOString()
    };
    // Persist to backend
    try {
      await axios.post(`${API_BASE_URL}/sessions`, {
        session_id: newSession.id,
        title: newSession.title
      });
    } catch (err) {
      console.error('Failed to create session on server:', err);
    }
    setSessions(prev => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    const shuffled = [...QUICK_QUERIES].sort(() => 0.5 - Math.random());
    setRandomQueries(shuffled.slice(0, 4));
    setSidebarOpen(false);
    return newSession.id;
  }, []);

  // Load sessions from backend when user logs in
  useEffect(() => {
    if (!user) {
      setSessions([]);
      setActiveSessionId(null);
      setSessionsLoaded(false);
      return;
    }
    setSessionsLoaded(false);
    axios.get(`${API_BASE_URL}/sessions`)
      .then(res => {
        const loaded = res.data.sessions || [];
        setSessions(loaded);
        if (loaded.length > 0) {
          setActiveSessionId(loaded[0].id);
          if (loaded[0].messages.length === 0) {
            const shuffled = [...QUICK_QUERIES].sort(() => 0.5 - Math.random());
            setRandomQueries(shuffled.slice(0, 4));
          }
        } else {
          createNewSession();
        }
        setSessionsLoaded(true);
      })
      .catch(err => {
        console.error('Failed to load sessions:', err);
        createNewSession();
        setSessionsLoaded(true);
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  useEffect(() => { scrollToBottom(); }, [sessions, activeSessionId, isLoading, scrollToBottom]);

  // Load active session soil report
  useEffect(() => {
    if (activeSessionId) {
      axios.get(`${API_BASE_URL}/soil-report/${activeSessionId}`)
        .then(res => setSoilReport(res.data.report || null))
        .catch(() => setSoilReport(null));
    } else {
      setSoilReport(null);
    }
  }, [activeSessionId]);

  /* ─── Soil Report Document Upload Handler ─── */
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setIsUploading(true);
    setStatus('Parsing Soil Test Document…');
    try {
      const resp = await axios.post(`${API_BASE_URL}/soil-report/upload?session_id=${activeSessionId}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setSoilReport(resp.data.report);
      const promptText = `I have uploaded my soil test report document: "${file.name}". Please analyze the extracted soil parameters, soil pH, NPK balance, and provide comprehensive fertilizer management recommendations.`;
      handleSend(promptText);
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to upload soil report document.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleRemoveSoilReport = async () => {
    try {
      await axios.delete(`${API_BASE_URL}/soil-report/${activeSessionId}`);
    } catch (_) {}
    setSoilReport(null);
  };

  const handleLogout = async () => {
    try {
      await firebaseSignOut(auth);
    } catch (e) {
      console.error('Logout error:', e);
    }
    localStorage.removeItem('agri_user');
    setUser(null);
  };

  if (authLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: 'var(--bg-primary)' }}>
        <div style={{ color: 'var(--text-secondary)', fontSize: '1rem' }}>Loading...</div>
      </div>
    );
  }

  if (!user) {
    return <AuthPage onLoginSuccess={(u) => setUser(u)} />;
  }

  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];
  const messages = activeSession?.messages || [];

  // Filtered sessions by search
  const filteredSessions = sessions.filter(s =>
    s.title.toLowerCase().includes(sessionSearch.toLowerCase())
  );

  /* ─── Session CRUD ─── */
  const deleteSession = async (id, e) => {
    e.stopPropagation();
    try { await axios.delete(`${API_BASE_URL}/sessions/${id}`); } catch (_) {}
    const next = sessions.filter(s => s.id !== id);
    setSessions(next);
    if (activeSessionId === id) {
      if (next.length > 0) setActiveSessionId(next[0].id);
      else createNewSession();
    }
  };

  const clearCurrentSession = async () => {
    if (!activeSessionId) return;
    if (window.confirm('Clear all messages in this consultation?')) {
      try { await axios.delete(`${API_BASE_URL}/sessions/${activeSessionId}/messages`); } catch (_) {}
      setSessions(prev => prev.map(s =>
        s.id === activeSessionId ? { ...s, messages: [], title: 'New Consultation' } : s
      ));
    }
  };

  const clearAllSessions = async () => {
    await Promise.all(sessions.map(s =>
      axios.delete(`${API_BASE_URL}/sessions/${s.id}`).catch(() => {})
    ));
    setSessions([]);
    createNewSession();
  };

  const updateActiveSession = (updates) => {
    setSessions(prev => prev.map(s =>
      s.id === activeSessionId ? { ...s, ...updates } : s
    ));
  };
  
  const updateSessionTitle = async (id, title) => {
    try { await axios.put(`${API_BASE_URL}/sessions/${id}`, { title }); } catch (_) {}
  };

  /* ─── Send ─── */
  const handleSend = async (userQuery) => {
    const q = userQuery || query;
    if (!q.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const isFirstMessage = messages.length === 0;
    const newTitle = isFirstMessage
      ? (q.length > 28 ? q.substring(0, 28) + '…' : q)
      : activeSession.title;

    const updatedMessages = [...messages, userMessage];
    updateActiveSession({
      messages: updatedMessages,
      title: newTitle
    });

    if (isFirstMessage) {
      updateSessionTitle(activeSessionId, newTitle);
    }

    setQuery('');
    setIsLoading(true);
    setStatus('Analyzing Context…');

    try {
      const resp = await axios.post(`${API_BASE_URL}/chat`, {
        query: q,
        session_id: activeSessionId
      });

      const assistantMessage = {
        role: 'assistant',
        content: resp.data.answer,
        followUps: resp.data.follow_ups || [],
        trace: resp.data.trace || null,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      updateActiveSession({ messages: [...updatedMessages, assistantMessage] });
    } catch (error) {
      const isAuth = error.response?.status === 401;
      if (isAuth) handleLogout();
      updateActiveSession({
        messages: [...updatedMessages, {
          role: 'assistant',
          content: isAuth
            ? '❌ **Session Expired**: Please log in again.'
            : '❌ **System Error**: Connection to AgriSense backend failed.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }]
      });
    } finally {
      setIsLoading(false);
      setStatus('');
    }
  };

  /* ─── Voice ─── */
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus' : 'audio/webm';
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType, audioBitsPerSecond: 128000 });
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        audioChunksRef.current = [];
        stream.getTracks().forEach(t => t.stop());
        await sendAudioForTranscription(blob);
      };
      mediaRecorderRef.current.start(1000);
      setIsRecording(true);
      setStatus('Listening...');
    } catch (err) {
      setStatus('Microphone access denied.');
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setStatus('Transcribing...');
    }
  };

  const sendAudioForTranscription = async (blob) => {
    setIsLoading(true);
    setStatus('Transcribing...');
    try {
      const formData = new FormData();
      formData.append('audio_file', blob, 'audio.webm');
      const resp = await axios.post(`${API_BASE_URL}/stt`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setQuery(resp.data.text);
      setStatus('');
      inputRef.current?.focus();
    } catch (_) {
      setStatus('Transcription failed.');
      setQuery('');
    } finally {
      setIsLoading(false);
    }
  };

  /* ─── Render ─── */
  return (
    <>
      {/* Settings Modal */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        theme={theme}
        setTheme={setTheme}
        onClearAllSessions={clearAllSessions}
      />

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      <div className="App">
        {/* ── SIDEBAR ── */}
        <aside className={`sidebar ${sidebarOpen ? 'mobile-open' : ''}`}>
          {/* Brand */}
          <div className="brand-box">
            <div className="brand-icon-wrap">
              <Sprout size={22} color="white" />
            </div>
            <div>
              <div className="brand-name">AgriSense</div>
              <div className="brand-sub">Scientific Guidance</div>
            </div>
          </div>

          {/* New Chat */}
          <button className="new-chat-btn" onClick={createNewSession}>
            <Plus size={17} />
            New Consultation
          </button>

          {/* Session Search */}
          <div className="session-search-wrap">
            <Search size={13} className="session-search-icon" />
            <input
              className="session-search-input"
              type="text"
              placeholder="Search sessions…"
              value={sessionSearch}
              onChange={e => setSessionSearch(e.target.value)}
            />
          </div>

          {filteredSessions.length > 0 && (
            <div className="sessions-label">Recent Sessions</div>
          )}

          {/* Session List */}
          <div className="sessions-list">
            {filteredSessions.map(s => (
              <div
                key={s.id}
                className={`session-item ${s.id === activeSessionId ? 'active' : ''}`}
                onClick={() => { setActiveSessionId(s.id); setSidebarOpen(false); }}
              >
                <MessageSquare size={13} />
                <div className="session-title">{s.title}</div>
                <button
                  className="delete-session-btn"
                  onClick={(e) => deleteSession(s.id, e)}
                  title="Delete session"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}

            {filteredSessions.length === 0 && sessionSearch && (
              <div style={{ padding: '0.5rem', color: 'rgba(255,255,255,0.3)', fontSize: '0.78rem', textAlign: 'center' }}>
                No sessions found
              </div>
            )}
          </div>

          {/* Bottom: User + Settings + Logout */}
          <div className="user-profile-section">
            <div className="user-info">
              <div className="user-avatar-small">
                {user?.avatar_url ? (
                  <img src={user.avatar_url} alt="User"
                    style={{ width: '100%', height: '100%', borderRadius: '8px' }}
                    referrerPolicy="no-referrer" />
                ) : (
                  <UserIcon size={15} />
                )}
              </div>
              <div className="user-details">
                <div className="user-name">{user.full_name || user.email.split('@')[0]}</div>
                <div className="user-email">{user.email}</div>
              </div>
            </div>
            <div className="user-actions">
              <button className="settings-btn" onClick={() => setSettingsOpen(true)} title="Settings">
                <Settings size={15} />
              </button>
              <button className="logout-btn" onClick={handleLogout} title="Log out">
                <LogOut size={15} />
              </button>
            </div>
          </div>

        </aside>

        {/* ── CHAT WINDOW ── */}
        <main className="chat-window">
          {/* Header */}
          <header className="chat-header">
            <div className="header-left">
              <button
                className="mobile-menu-btn"
                onClick={() => setSidebarOpen(true)}
                aria-label="Open sidebar"
              >
                <Menu size={18} />
              </button>
              <div className="header-title">
                {activeSession?.title || 'New Consultation'}
                {isLoading && <span className="live-badge">● Live Analysis</span>}
              </div>
              <div className="header-meta">
                <span>AgriSense Precision AI</span>
                <span className="header-dot" />
                <span>Session {activeSessionId?.slice(-4)}</span>
              </div>
            </div>

            <div className="header-actions">
              {/* Theme Toggle */}
              <button
                className="theme-toggle-btn"
                onClick={toggleTheme}
                title={theme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode'}
              >
                {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
              </button>

              {/* Clear */}
              <button className="icon-btn" onClick={clearCurrentSession} title="Clear conversation">
                <Trash2 size={16} />
              </button>
            </div>
          </header>

          {/* ── MESSAGES ── */}
          <div className="messages-container">

            {/* Welcome Screen */}
            {messages.length === 0 && (
              <div className="welcome-screen">
                <div className="welcome-icon-ring">
                  <Sprout size={46} color={theme === 'dark' ? '#74c69d' : '#1a5c3a'} />
                </div>
                <h2 className="welcome-title">Welcome to AgriSense AI</h2>
                <p className="welcome-sub">
                  Hello {user?.full_name || 'there'}! I'm your scientific agricultural assistant.
                  I can help with soil health, pest control, and regional crop selection.
                </p>

                <div className="welcome-message-card">
                  <p>How can I help you grow today? Try one of these suggestions or type your query below.</p>
                </div>

                <div className="quick-grid">
                  {randomQueries.map(q => (
                    <button key={q} onClick={() => handleSend(q)} className="quick-card">
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Message List */}
            {messages.map((msg, index) => (
              <div key={index} className={`msg-row ${msg.role}`}>
                <div className="msg-with-avatar">
                  {/* Avatar */}
                  <div className={`avatar ${msg.role === 'assistant' ? 'ai-avatar' : 'user-avatar'}`}>
                    {msg.role === 'assistant' ? (
                      <Leaf size={14} />
                    ) : (
                      user?.avatar_url ? (
                        <img src={user.avatar_url} alt="User"
                          style={{ width: '100%', height: '100%', borderRadius: '50%' }}
                          referrerPolicy="no-referrer" />
                      ) : 'U'
                    )}
                  </div>

                  {/* Bubble */}
                  <div className="bubble-wrapper">
                    <div className="bubble">
                      <div className="markdown-body">
                        <ReactMarkdown>
                          {msg.role === 'assistant' ? cleanAgriResponse(msg.content) : msg.content}
                        </ReactMarkdown>
                      </div>
                      {msg.role === 'assistant' && <TraceBlock trace={msg.trace} />}
                    </div>
                    {/* Actions for assistant messages */}
                    {msg.role === 'assistant' && (
                      <div className="msg-actions" style={{display: 'flex', gap: '5px', alignSelf: 'flex-start', marginTop: '5px'}}>
                        <CopyButton text={msg.content} />
                        <AudioButton text={cleanAgriResponse(msg.content)} />
                      </div>
                    )}
                  </div>
                </div>

                {/* Footer */}
                <div className="msg-footer">
                  <span>{msg.timestamp}</span>
                  {msg.role === 'assistant' && (
                    <span className="verified-badge">
                      <CheckCircle2 size={10} /> Verified Data
                    </span>
                  )}
                </div>

                {/* Follow-ups */}
                {msg.role === 'assistant' && msg.followUps?.length > 0 && (
                  <div className="follow-ups">
                    {msg.followUps.map((f, i) => (
                      <button key={i} className="follow-up-cell" onClick={() => handleSend(f)}>
                        <ChevronRight size={13} />
                        {f}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {/* Typing Indicator */}
            {isLoading && <TypingIndicator status={status} />}
            <div ref={messagesEndRef} />
          </div>

          {/* ── INPUT ── */}
          <div className="input-section">
            {soilReport && (
              <div className="soil-report-badge-bar">
                <FileText size={15} className="badge-icon" />
                <span>Soil Test Report Attached: <strong>{soilReport.filename}</strong></span>
                <button type="button" onClick={handleRemoveSoilReport} className="remove-report-btn" title="Remove attached soil report">
                  <X size={14} />
                </button>
              </div>
            )}

            <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="input-bar">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".pdf,.png,.jpg,.jpeg,.webp"
                style={{ display: 'none' }}
              />

              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="send-button upload-btn"
                title="Upload Soil Test Report (PDF or Image)"
                disabled={isLoading || isUploading || isRecording}
              >
                {isUploading ? <UploadCloud size={19} className="spin-icon" /> : <Paperclip size={19} />}
              </button>

              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={isUploading ? 'Parsing soil document…' : isRecording ? 'Recording… speak now' : 'Type your agricultural query or upload a soil report…'}
                disabled={isLoading || isUploading || isRecording}
              />

              {isRecording ? (
                <button type="button" onClick={stopRecording} className="send-button recording">
                  <StopCircle size={19} />
                </button>
              ) : (
                <button type="button" onClick={startRecording} className="send-button" disabled={isLoading || isUploading}>
                  <Mic size={19} />
                </button>
              )}

              <button type="submit" disabled={isLoading || isUploading || !query.trim() || isRecording} className="send-button">
                <Send size={19} />
              </button>
            </form>

            <div className="input-footer">
              <span className="powered-by">Powered by Llama-3.3-70B</span>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}

function App() {
  return <AppContent />;
}

export default App;