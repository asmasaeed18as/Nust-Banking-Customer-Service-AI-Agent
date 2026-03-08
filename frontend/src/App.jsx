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

/* ─── Simple Markdown Renderer ────────────────────────────────────────────── */
function renderMarkdown(text) {
  if (!text) return []
  const lines = text.split('\n')
  const elements = []
  let key = 0

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    // Numbered list
    if (/^\d+\.\s/.test(line)) {
      elements.push(<div key={key++} className="md-list-item numbered">
        <span className="md-num">{line.match(/^(\d+\.)/)[1]}</span>
        <span dangerouslySetInnerHTML={{ __html: inlineFormat(line.replace(/^\d+\.\s/, '')) }} />
      </div>)
    }
    // Bullet list
    else if (/^[-•·o]\s/.test(line)) {
      elements.push(<div key={key++} className="md-list-item">
        <span className="md-bullet">•</span>
        <span dangerouslySetInnerHTML={{ __html: inlineFormat(line.replace(/^[-•·o]\s/, '')) }} />
      </div>)
    }
    // Empty line → spacing
    else if (line.trim() === '') {
      elements.push(<div key={key++} className="md-spacer" />)
    }
    // Normal text
    else {
      elements.push(<p key={key++} dangerouslySetInnerHTML={{ __html: inlineFormat(line) }} />)
    }
  }
  return elements
}

function inlineFormat(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
}

/* ─── Message Bubble ───────────────────────────────────────────────────────── */
function MessageBubble({ msg }) {
  const isUser = msg.role === 'user'
  const isBlocked = msg.blocked
  const [copied, setCopied] = React.useState(false)

  const copyText = () => {
    navigator.clipboard.writeText(msg.content).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className={`message-row ${isUser ? 'user' : ''}`}>
      {!isUser && <div className="avatar bot">🏦</div>}
      <div className="bubble-wrap">
        <div className={`bubble ${isUser ? 'user' : 'bot'} ${isBlocked ? 'blocked' : ''}`}>
          {isBlocked && <span>⛔ </span>}
          {isUser
            ? msg.content
            : <div className="md-body">{renderMarkdown(msg.content)}</div>
          }
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

        <div className="msg-meta">
          <span className="msg-time">{formatTime(msg.time)}</span>
          {!isUser && msg.latency && (
            <span className="msg-latency">{(msg.latency / 1000).toFixed(1)}s</span>
          )}
          {!isUser && (
            <button className="copy-btn" onClick={copyText} title="Copy response">
              {copied ? '✅' : '📋'}
            </button>
          )}
        </div>
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

/* ─── User Profile Panel ─────────────────────────────────────────────── */
const EXISTING_PRODUCTS = [
  'Savings Account', 'Current Account', 'Term Deposit',
  'Debit Card', 'Credit Card', 'Personal Finance', 'Home Finance', 'Auto Finance'
]
const INTEREST_OPTIONS = [
  'Account Opening', 'Loans & Finance', 'Cards',
  'Transfers & Payments', 'Investments', 'Mobile Banking', 'Remittances'
]

function ProfilePanel({ profile, setProfile }) {
  const toggle = (field, val) => setProfile(prev => {
    const arr = prev[field]
    return { ...prev, [field]: arr.includes(val) ? arr.filter(x => x !== val) : [...arr, val] }
  })

  const hasData = profile.customerType || profile.employment || profile.ageGroup ||
    profile.existingProducts.length > 0 || profile.interests.length > 0

  return (
    <div className="profile-panel">
      <div className="profile-header">
        <div className="profile-icon">👤</div>
        <div>
          <h2>Your Profile</h2>
          <p>Optional — helps the AI give more relevant answers.</p>
        </div>
      </div>

      {hasData && (
        <div className="profile-active-badge">✅ Profile active — AI will use your context</div>
      )}

      <div className="profile-form">

        <div className="profile-section">
          <label className="profile-label">Customer Type</label>
          <select
            className="profile-select"
            value={profile.customerType}
            onChange={e => setProfile(p => ({ ...p, customerType: e.target.value }))}
          >
            <option value="">Not specified</option>
            <option>Individual</option>
            <option>Business</option>
            <option>Non-Resident Pakistani (NRP)</option>
            <option>Student</option>
          </select>
        </div>

        <div className="profile-section">
          <label className="profile-label">Employment Status</label>
          <select
            className="profile-select"
            value={profile.employment}
            onChange={e => setProfile(p => ({ ...p, employment: e.target.value }))}
          >
            <option value="">Not specified</option>
            <option>Salaried</option>
            <option>Self-Employed</option>
            <option>Business Owner</option>
            <option>Retired</option>
            <option>Student</option>
          </select>
        </div>

        <div className="profile-section">
          <label className="profile-label">Age Group</label>
          <select
            className="profile-select"
            value={profile.ageGroup}
            onChange={e => setProfile(p => ({ ...p, ageGroup: e.target.value }))}
          >
            <option value="">Not specified</option>
            <option>Under 25</option>
            <option>25–40</option>
            <option>40–55</option>
            <option>55+</option>
          </select>
        </div>

        <div className="profile-section">
          <label className="profile-label">Existing Products <span className="profile-hint">(select all that apply)</span></label>
          <div className="profile-checks">
            {EXISTING_PRODUCTS.map(p => (
              <label key={p} className="check-item">
                <input
                  type="checkbox"
                  checked={profile.existingProducts.includes(p)}
                  onChange={() => toggle('existingProducts', p)}
                />
                {p}
              </label>
            ))}
          </div>
        </div>

        <div className="profile-section">
          <label className="profile-label">Areas of Interest <span className="profile-hint">(select all that apply)</span></label>
          <div className="profile-checks">
            {INTEREST_OPTIONS.map(p => (
              <label key={p} className="check-item">
                <input
                  type="checkbox"
                  checked={profile.interests.includes(p)}
                  onChange={() => toggle('interests', p)}
                />
                {p}
              </label>
            ))}
          </div>
        </div>

        {hasData && (
          <button
            className="profile-clear-btn"
            onClick={() => setProfile({ customerType: '', employment: '', ageGroup: '', existingProducts: [], interests: [] })}
          >
            Clear Profile
          </button>
        )}
      </div>
    </div>
  )
}

/* ─── Main App ─────────────────────────────────────────────────────────────── */
export default function App() {
  const [view, setView] = useState('chat')       // 'chat' | 'upload' | 'profile'
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [serverStatus, setServerStatus] = useState('loading')
  const [userProfile, setUserProfile] = useState({
    customerType: '', employment: '', ageGroup: '',
    existingProducts: [], interests: []
  })
  const bottomRef = useRef()
  const textareaRef = useRef()
  // Stable session ID — one per page load, unique per browser tab
  const sessionId = useRef(crypto.randomUUID()).current

  // Build a compact user_context object — only include non-empty fields
  const buildUserContext = () => {
    const ctx = {}
    if (userProfile.customerType) ctx.customer_type = userProfile.customerType
    if (userProfile.employment) ctx.employment = userProfile.employment
    if (userProfile.ageGroup) ctx.age_group = userProfile.ageGroup
    if (userProfile.existingProducts.length) ctx.existing_products = userProfile.existingProducts
    if (userProfile.interests.length) ctx.interests = userProfile.interests
    return Object.keys(ctx).length ? ctx : null
  }

  const hasProfile = buildUserContext() !== null

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
      const payload = { query: text, session_id: sessionId }
      const ctx = buildUserContext()
      if (ctx) payload.user_context = ctx

      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
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

  const clearChat = () => {
    setMessages([])
    // Also wipe server-side history so the model forgets previous turns
    fetch(`${API}/chat/clear?session_id=${sessionId}`, { method: 'POST' }).catch(() => { })
  }

  // ── Sidebar nav items ──────────────────────────────────────────────────────
  const navItems = [
    { id: 'chat', icon: '💬', label: 'AI Assistant' },
    { id: 'profile', icon: '👤', label: 'My Profile' },
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
            <h1>
              {view === 'chat' ? 'AI Banking Assistant' : view === 'upload' ? 'Document Upload' : 'My Profile'}
            </h1>
            <p>
              {view === 'chat'
                ? <>
                  Powered by NUST Bank knowledge base · Qwen 2.5-3B
                  {hasProfile && <span className="profile-badge">👤 Profile active</span>}
                </>
                : view === 'upload'
                  ? 'Add new knowledge to the AI in real-time'
                  : 'Optional context to personalise AI responses'
              }
            </p>
          </div>
          {view === 'chat' && (
            <div className="header-actions">
              <button className="icon-btn" title="My Profile" onClick={() => setView('profile')}>
                {hasProfile ? '👤' : '👥'}
              </button>
              <button className="icon-btn" title="Clear Chat" onClick={clearChat}>🗑️</button>
            </div>
          )}
        </header>

        {view === 'upload' ? (
          <UploadPanel />
        ) : view === 'profile' ? (
          <ProfilePanel profile={userProfile} setProfile={setUserProfile} />
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
