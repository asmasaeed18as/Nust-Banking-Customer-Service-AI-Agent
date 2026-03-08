import React, { useState, useRef, useCallback } from 'react'

const API = 'http://localhost:8000/api'

const QUICK_CHIPS = [
  'What is my daily transfer limit?',
  'How do I reset my MPIN?',
  'Can I bank while overseas?',
  'What is home remittance?',
  'How to add a new beneficiary?',
]

function formatTime(date) {
  return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })
}

/* ─── Message Bubble ───────────────────────────────────────────────────────── */
function MessageBubble({ msg }) {
  const isUser = msg.role === 'user'
  const isBlocked = msg.blocked

  return (
    <div className={`message-row ${isUser ? 'user' : ''}`}>
      {!isUser && <div className="avatar bot">🏦</div>}
      <div className="bubble-wrap">
        <div className={`bubble ${isUser ? 'user' : 'bot'} ${isBlocked ? 'blocked' : ''}`}>
          {isBlocked && <span>⛔ </span>}
          {msg.content}
        </div>

        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="sources-bar">
            {msg.sources.slice(0, 3).map((s, i) => (
              <span key={i} className="source-tag">
                📄 {s.source.replace('Excel - ', '').replace('JSON FAQ', 'App FAQ')}
                <span className="source-score">{(s.score * 100).toFixed(0)}%</span>
              </span>
            ))}
          </div>
        )}

        <span className="msg-time">{formatTime(msg.time)}</span>
      </div>
      {isUser && <div className="avatar user">👤</div>}
    </div>
  )
}

/* ─── Typing Indicator ─────────────────────────────────────────────────────── */
function TypingIndicator() {
  return (
    <div className="message-row">
      <div className="avatar bot">🏦</div>
      <div className="bubble bot" style={{ padding: '14px 18px' }}>
        <div className="typing-indicator">
          <span /><span /><span />
        </div>
      </div>
    </div>
  )
}

/* ─── Upload Panel ─────────────────────────────────────────────────────────── */
function UploadPanel() {
  const [status, setStatus] = useState(null)  // null | 'ok' | 'error'
  const [msg, setMsg] = useState('')
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef()

  const handleFile = async (file) => {
    if (!file) return
    const fd = new FormData()
    fd.append('file', file)
    setStatus(null)
    setMsg('Uploading and indexing...')
    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: fd })
      const data = await res.json()
      if (res.ok) { setStatus('ok'); setMsg(`✅ ${data.message}`) }
      else { setStatus('error'); setMsg(`❌ ${data.detail}`) }
    } catch (e) {
      setStatus('error')
      setMsg('❌ Could not connect to backend.')
    }
  }

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  return (
    <div className="upload-panel">
      <div
        className={`upload-card ${dragging ? 'drag-over' : ''}`}
        onClick={() => fileRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div className="upload-icon">📂</div>
        <h3>Upload Bank Document</h3>
        <p>Add new FAQs, policies, or product information.<br />The AI will learn from it instantly.</p>
        <div className="upload-formats">
          <span className="format-badge">JSON</span>
          <span className="format-badge">CSV</span>
          <span className="format-badge">TXT</span>
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".json,.csv,.txt"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>
      {msg && <div className={`upload-result ${status === 'error' ? 'error' : ''}`}>{msg}</div>}
    </div>
  )
}

/* ─── Main App ─────────────────────────────────────────────────────────────── */
export default function App() {
  const [view, setView] = useState('chat')       // 'chat' | 'upload'
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [serverStatus, setServerStatus] = useState('loading')
  const bottomRef = useRef()
  const textareaRef = useRef()

  // ── Health check on mount ──────────────────────────────────────────────────
  React.useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.ok ? setServerStatus('online') : setServerStatus('error'))
      .catch(() => setServerStatus('error'))
  }, [])

  // ── Auto-scroll ────────────────────────────────────────────────────────────
  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // ── Send message ───────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (query) => {
    const text = (query || input).trim()
    if (!text || loading) return

    setMessages(prev => [...prev, {
      role: 'user', content: text, time: new Date()
    }])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'bot',
        content: data.answer,
        sources: data.sources,
        blocked: data.blocked,
        latency: data.latency_ms,
        time: new Date(),
      }])
    } catch {
      setMessages(prev => [...prev, {
        role: 'bot',
        content: '⚠️ Could not reach the NUST Bank server. Please ensure the backend is running.',
        time: new Date(),
      }])
    } finally {
      setLoading(false)
    }
  }, [input, loading])

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  const clearChat = () => setMessages([])

  // ── Sidebar nav items ──────────────────────────────────────────────────────
  const navItems = [
    { id: 'chat', icon: '💬', label: 'AI Assistant' },
    { id: 'upload', icon: '📤', label: 'Upload Document' },
  ]

  return (
    <div className="app-shell">

      {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon">🏦</div>
          <div className="logo-text">
            <h2>NUST Bank</h2>
            <span>AI Customer Service</span>
          </div>
        </div>

        <div className="sidebar-section-title">Navigation</div>
        {navItems.map(item => (
          <button
            key={item.id}
            className={`sidebar-btn ${view === item.id ? 'active' : ''}`}
            onClick={() => setView(item.id)}
          >
            <span className="btn-icon">{item.icon}</span>
            {item.label}
          </button>
        ))}

        <div className="sidebar-footer">
          <div className="status-pill">
            <div className={`status-dot ${serverStatus === 'online' ? '' : serverStatus === 'loading' ? 'loading' : 'error'}`} />
            {serverStatus === 'online' ? 'AI Online' : serverStatus === 'loading' ? 'Connecting…' : 'Server Offline'}
          </div>
        </div>
      </aside>

      {/* ── Chat Main ───────────────────────────────────────────────────────── */}
      <main className="chat-main">
        <header className="chat-header">
          <div className="chat-header-info">
            <h1>{view === 'chat' ? 'AI Banking Assistant' : 'Document Upload'}</h1>
            <p>{view === 'chat' ? 'Powered by NUST Bank knowledge base · Qwen 2.5-3B' : 'Add new knowledge to the AI in real-time'}</p>
          </div>
          {view === 'chat' && (
            <div className="header-actions">
              <button className="icon-btn" title="Clear Chat" onClick={clearChat}>🗑️</button>
            </div>
          )}
        </header>

        {view === 'upload' ? (
          <UploadPanel />
        ) : (
          <>
            <div className="messages-area">
              {messages.length === 0 ? (
                <div className="welcome-screen">
                  <div className="welcome-icon">🏦</div>
                  <h2>How can I help you today?</h2>
                  <p>Ask me anything about NUST Bank — accounts, transfers, loans, and more. I'm here to help 24/7.</p>
                  <div className="quick-chips">
                    {QUICK_CHIPS.map((c, i) => (
                      <button key={i} className="chip" onClick={() => sendMessage(c)}>{c}</button>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)
              )}
              {loading && <TypingIndicator />}
              <div ref={bottomRef} />
            </div>

            <div className="input-bar">
              <div className="input-wrap">
                <textarea
                  ref={textareaRef}
                  rows={1}
                  placeholder="Ask about accounts, transfers, loans…"
                  value={input}
                  onChange={(e) => {
                    setInput(e.target.value)
                    e.target.style.height = 'auto'
                    e.target.style.height = e.target.scrollHeight + 'px'
                  }}
                  onKeyDown={handleKey}
                  disabled={loading}
                />
                <button
                  className="send-btn"
                  onClick={() => sendMessage()}
                  disabled={!input.trim() || loading}
                  title="Send (Enter)"
                >
                  ↑
                </button>
              </div>
              <p className="input-hint">Press Enter to send · Shift+Enter for new line</p>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
