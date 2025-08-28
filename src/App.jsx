import React, { useState } from 'react'

const API_BASE = 'http://127.0.0.1:8000'

export default function App() {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hi there! I can track orders, explain returns/refunds, product info, or escalate to a human.' }
  ])
  const [input, setInput] = useState('')

  async function sendMessage() {
    const text = input.trim()
    if (!text) return
    setMessages(prev => [...prev, { sender: 'user', text }])
    setInput('')

    try {
      const r = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, user_id: 'web-user', conversation_id: 'conv-web' })
      })
      const data = await r.json()
      setMessages(prev => [...prev, { sender: 'bot', text: data.reply }])
    } catch (e) {
      setMessages(prev => [...prev, { sender: 'bot', text: 'Network error. Is the backend running on :8000?' }])
    }
  }

  function onKey(e) {
    if (e.key === 'Enter') sendMessage()
  }

  return (
    <div style={{ maxWidth: 680, margin: '40px auto', fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial' }}>
      <h1>Customer Service Chatbot</h1>
      <div style={{ border: '1px solid #ddd', borderRadius: 12, padding: 16, minHeight: 360 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ textAlign: m.sender === 'user' ? 'right' : 'left', margin: '8px 0' }}>
            <span style={{
              display: 'inline-block',
              padding: '10px 14px',
              borderRadius: 14,
              background: m.sender === 'user' ? '#e6f2ff' : '#f5f5f5'
            }}>{m.text}</span>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="Type a message…"
          style={{ flex: 1, padding: 12, borderRadius: 12, border: '1px solid #ddd' }}
        />
        <button onClick={sendMessage} style={{ padding: '12px 16px', borderRadius: 12, border: '1px solid #ddd', cursor: 'pointer' }}>
          Send
        </button>
      </div>
    </div>
  )
}
