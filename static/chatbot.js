/**
 * chatbot.js — PhysBot floating chat widget
 * Semiconductor Physics RAG chatbot UI
 * Features: floating FAB, slide-up panel, markdown rendering,
 * source attribution, typing indicator, conversation history,
 * suggested questions, context word counter.
 */

(function () {
  'use strict';

  // ── Inject styles ──────────────────────────────────────────
  const style = document.createElement('style');
  style.textContent = `
    /* ── FAB Button ── */
    #physbot-fab {
      position: fixed; bottom: 28px; right: 28px; z-index: 9999;
      width: 58px; height: 58px; border-radius: 50%;
      background: linear-gradient(135deg, #00f0ff, #9b5de5);
      border: none; cursor: pointer;
      box-shadow: 0 4px 20px #00f0ff55, 0 0 0 0 #00f0ff44;
      display: flex; align-items: center; justify-content: center;
      transition: transform 0.2s, box-shadow 0.2s;
      animation: fab-pulse 3s ease-in-out infinite;
    }
    #physbot-fab:hover {
      transform: scale(1.1);
      box-shadow: 0 6px 28px #00f0ff88, 0 0 0 8px #00f0ff22;
    }
    #physbot-fab svg { width: 26px; height: 26px; }
    @keyframes fab-pulse {
      0%,100% { box-shadow: 0 4px 20px #00f0ff55, 0 0 0 0 #00f0ff00; }
      50%      { box-shadow: 0 4px 20px #00f0ff88, 0 0 0 10px #00f0ff00; }
    }
    .physbot-badge {
      position: absolute; top: -4px; right: -4px;
      width: 18px; height: 18px; border-radius: 50%;
      background: #ff3e8a; color: #fff;
      font-size: 10px; font-weight: 700;
      display: flex; align-items: center; justify-content: center;
      font-family: 'Space Mono', monospace;
      border: 2px solid #050810;
      transition: opacity 0.3s;
    }

    /* ── Panel ── */
    #physbot-panel {
      position: fixed; bottom: 100px; right: 28px; z-index: 9998;
      width: 400px; max-width: calc(100vw - 40px);
      height: 560px; max-height: calc(100vh - 120px);
      background: #0a0f1a;
      border: 1px solid #1a2840;
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.8), 0 0 0 1px #00f0ff18;
      display: flex; flex-direction: column;
      transform: translateY(20px) scale(0.95);
      opacity: 0; pointer-events: none;
      transition: transform 0.25s cubic-bezier(0.34,1.56,0.64,1),
                  opacity 0.2s ease;
      font-family: 'DM Sans', sans-serif;
      overflow: hidden;
    }
    #physbot-panel.open {
      transform: translateY(0) scale(1);
      opacity: 1; pointer-events: all;
    }

    /* ── Header ── */
    .pb-header {
      padding: 14px 16px 12px;
      background: #0d1117;
      border-bottom: 1px solid #1a2840;
      display: flex; align-items: center; gap: 10px;
      flex-shrink: 0;
    }
    .pb-avatar {
      width: 34px; height: 34px; border-radius: 50%;
      background: linear-gradient(135deg, #00f0ff22, #9b5de522);
      border: 1px solid #00f0ff44;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
    }
    .pb-avatar svg { width: 18px; height: 18px; }
    .pb-info { flex: 1; }
    .pb-name {
      font-family: 'Space Mono', monospace; font-size: 0.75rem;
      font-weight: 700; color: #00f0ff; letter-spacing: 0.08em;
    }
    .pb-status {
      font-size: 0.62rem; color: #6a8aaa; margin-top: 1px;
      display: flex; align-items: center; gap: 4px;
    }
    .pb-status-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: #00ff9d; box-shadow: 0 0 6px #00ff9d;
      animation: status-blink 2s infinite;
    }
    @keyframes status-blink {
      0%,100%{opacity:1} 50%{opacity:0.4}
    }
    .pb-close {
      background: none; border: none; cursor: pointer;
      color: #3d5470; font-size: 18px; padding: 4px;
      border-radius: 6px; transition: color 0.2s, background 0.2s;
      line-height: 1;
    }
    .pb-close:hover { color: #ff3e8a; background: #ff3e8a18; }

    /* ── Source bar ── */
    .pb-sources {
      padding: 6px 12px;
      background: #080c10;
      border-bottom: 1px solid #1a2840;
      display: flex; align-items: center; gap: 6px;
      flex-wrap: wrap; flex-shrink: 0;
    }
    .pb-src-label {
      font-family: 'Space Mono', monospace; font-size: 0.55rem;
      color: #2e4460; letter-spacing: 0.1em; text-transform: uppercase;
    }
    .pb-src-badge {
      font-family: 'Space Mono', monospace; font-size: 0.58rem;
      padding: 2px 7px; border-radius: 10px;
      border: 1px solid #1a2840; color: #6a8aaa;
      background: #0d1117;
      transition: border-color 0.2s, color 0.2s;
    }
    .pb-src-badge.active { border-color: #00f0ff55; color: #00f0ff; }

    /* ── Messages ── */
    .pb-messages {
      flex: 1; overflow-y: auto; padding: 14px 14px 8px;
      display: flex; flex-direction: column; gap: 12px;
    }
    .pb-messages::-webkit-scrollbar { width: 3px; }
    .pb-messages::-webkit-scrollbar-thumb { background: #1a2840; border-radius: 2px; }

    .pb-msg { display: flex; gap: 8px; align-items: flex-start; }
    .pb-msg.user { flex-direction: row-reverse; }

    .pb-bubble {
      max-width: 82%; padding: 9px 13px; border-radius: 12px;
      font-size: 0.82rem; line-height: 1.65; word-break: break-word;
    }
    .pb-msg.bot .pb-bubble {
      background: #0f1624; border: 1px solid #1a2840;
      color: #d8eaff; border-radius: 4px 12px 12px 12px;
    }
    .pb-msg.user .pb-bubble {
      background: linear-gradient(135deg, #00f0ff18, #9b5de518);
      border: 1px solid #00f0ff33; color: #d8eaff;
      border-radius: 12px 4px 12px 12px;
    }

    /* Markdown in bubbles */
    .pb-bubble strong { color: #00f0ff; font-weight: 700; }
    .pb-bubble em { color: #ffb800; font-style: normal; }
    .pb-bubble code {
      font-family: 'Space Mono', monospace; font-size: 0.78rem;
      background: #050810; color: #00ff9d;
      padding: 1px 5px; border-radius: 4px;
      border: 1px solid #1a2840;
    }
    .pb-bubble pre {
      background: #050810; border: 1px solid #1a2840;
      border-radius: 8px; padding: 10px 12px; margin: 6px 0;
      overflow-x: auto;
    }
    .pb-bubble pre code {
      background: none; border: none; padding: 0;
      font-size: 0.75rem; color: #00ff9d;
    }
    .pb-bubble ul, .pb-bubble ol {
      margin: 4px 0; padding-left: 18px;
    }
    .pb-bubble li { margin: 2px 0; }
    .pb-bubble p { margin: 0 0 6px; }
    .pb-bubble p:last-child { margin-bottom: 0; }

    /* Source attribution inside bubble */
    .pb-attribution {
      margin-top: 6px; padding-top: 6px;
      border-top: 1px solid #1a2840;
      font-family: 'Space Mono', monospace;
      font-size: 0.58rem; color: #2e4460;
      display: flex; flex-wrap: wrap; gap: 4px; align-items: center;
    }
    .pb-attr-tag {
      padding: 1px 6px; border-radius: 8px;
      border: 1px solid #1a2840; color: #4a6a8a;
    }
    .pb-ctx-count { margin-left: auto; color: #2e4460; }

    /* Mini avatar */
    .pb-mini-av {
      width: 24px; height: 24px; border-radius: 50%; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      font-family: 'Space Mono', monospace; font-size: 0.6rem;
      margin-top: 2px;
    }
    .pb-msg.bot  .pb-mini-av { background: #00f0ff18; border: 1px solid #00f0ff33; color: #00f0ff; }
    .pb-msg.user .pb-mini-av { background: #9b5de518; border: 1px solid #9b5de533; color: #9b5de5; }

    /* Typing indicator */
    .pb-typing {
      display: flex; gap: 4px; padding: 8px 12px;
      align-items: center;
    }
    .pb-typing span {
      width: 6px; height: 6px; border-radius: 50%;
      background: #00f0ff; opacity: 0.4;
      animation: typing-dot 1.2s ease-in-out infinite;
    }
    .pb-typing span:nth-child(2) { animation-delay: 0.2s; }
    .pb-typing span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing-dot {
      0%,60%,100%{opacity:0.2;transform:scale(1)}
      30%{opacity:1;transform:scale(1.3)}
    }

    /* ── Suggestions ── */
    .pb-suggestions {
      padding: 6px 10px 4px;
      display: flex; gap: 5px; flex-wrap: wrap; flex-shrink: 0;
      border-top: 1px solid #1a2840;
      background: #080c10;
    }
    .pb-sugg {
      font-family: 'Space Mono', monospace; font-size: 0.58rem;
      padding: 4px 9px; border-radius: 12px;
      border: 1px solid #1a2840; color: #6a8aaa;
      background: #0d1117; cursor: pointer;
      transition: all 0.18s; white-space: nowrap;
    }
    .pb-sugg:hover { border-color: #00f0ff55; color: #00f0ff; background: #00f0ff08; }

    /* ── Input bar ── */
    .pb-input-bar {
      padding: 10px 12px;
      border-top: 1px solid #1a2840;
      background: #0d1117;
      display: flex; gap: 8px; align-items: flex-end;
      flex-shrink: 0;
    }
    #physbot-input {
      flex: 1; background: #080c10;
      border: 1px solid #1a2840; border-radius: 10px;
      color: #d8eaff; font-family: 'DM Sans', sans-serif;
      font-size: 0.82rem; padding: 8px 12px;
      resize: none; outline: none; min-height: 36px; max-height: 100px;
      transition: border-color 0.2s;
    }
    #physbot-input:focus { border-color: #00f0ff55; }
    #physbot-input::placeholder { color: #2e4460; }
    #physbot-send {
      width: 36px; height: 36px; border-radius: 10px;
      background: linear-gradient(135deg, #00f0ff22, #9b5de522);
      border: 1px solid #00f0ff44; color: #00f0ff;
      cursor: pointer; display: flex; align-items: center;
      justify-content: center; flex-shrink: 0;
      transition: all 0.18s;
    }
    #physbot-send:hover {
      background: linear-gradient(135deg, #00f0ff44, #9b5de544);
      box-shadow: 0 0 12px #00f0ff44;
    }
    #physbot-send svg { width: 16px; height: 16px; }

    /* Clear button */
    .pb-clear {
      background: none; border: none; cursor: pointer;
      color: #2e4460; font-size: 11px;
      font-family: 'Space Mono', monospace;
      padding: 4px 6px; border-radius: 4px;
      transition: color 0.2s;
    }
    .pb-clear:hover { color: #ff3e8a; }

    /* ── Responsive ── */
    @media (max-width: 480px) {
      #physbot-panel { width: calc(100vw - 20px); right: 10px; bottom: 90px; }
      #physbot-fab   { right: 18px; bottom: 18px; }
    }
  `;
  document.head.appendChild(style);

  // ── State ──────────────────────────────────────────────────
  let isOpen    = false;
  let history   = [];
  let isLoading = false;
  let unread    = 0;

  const SUGGESTIONS = [
    "What is the Fermi-Dirac distribution?",
    "Explain Brillouin zones",
    "How does MOSFET work?",
    "Kronig-Penney model",
    "Drude model & mobility",
    "p-n junction physics",
    "What is CMOS?",
    "Phonon dispersion",
    "Density of states",
  ];

  // ── Build DOM ──────────────────────────────────────────────
  function buildUI() {
    // FAB
    const fab = document.createElement('button');
    fab.id = 'physbot-fab';
    fab.setAttribute('aria-label', 'Open PhysBot chat');
    fab.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
           style="color:#fff">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        <circle cx="9" cy="10" r="1" fill="white" stroke="none"/>
        <circle cx="12" cy="10" r="1" fill="white" stroke="none"/>
        <circle cx="15" cy="10" r="1" fill="white" stroke="none"/>
      </svg>
      <span class="physbot-badge" id="pb-badge" style="opacity:0">0</span>
    `;
    document.body.appendChild(fab);

    // Panel
    const panel = document.createElement('div');
    panel.id = 'physbot-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'PhysBot semiconductor physics assistant');
    panel.innerHTML = `
      <!-- Header -->
      <div class="pb-header">
        <div class="pb-avatar">
          <svg viewBox="0 0 24 24" fill="none" stroke="#00f0ff" stroke-width="2">
            <circle cx="12" cy="12" r="3"/>
            <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83
                     M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
          </svg>
        </div>
        <div class="pb-info">
          <div class="pb-name">PHYSBOT</div>
          <div class="pb-status">
            <span class="pb-status-dot"></span>
            <span id="pb-status-text">Live RAG · Wikipedia + ArXiv</span>
          </div>
        </div>
        <button class="pb-clear" id="pb-clear-btn" title="Clear conversation">CLR</button>
        <button class="pb-close" id="pb-close-btn" aria-label="Close">✕</button>
      </div>

      <!-- Source bar -->
      <div class="pb-sources">
        <span class="pb-src-label">Sources:</span>
        <span class="pb-src-badge active" id="src-wiki">Wikipedia</span>
        <span class="pb-src-badge active" id="src-arxiv">ArXiv</span>
        <span class="pb-src-badge" id="src-ollama">llama3</span>
        <span style="margin-left:auto;font-family:'Space Mono',monospace;font-size:0.55rem;color:#2e4460" id="pb-ctx-info"></span>
      </div>

      <!-- Messages -->
      <div class="pb-messages" id="pb-messages"></div>

      <!-- Suggestions -->
      <div class="pb-suggestions" id="pb-suggestions"></div>

      <!-- Input -->
      <div class="pb-input-bar">
        <textarea id="physbot-input" rows="1"
          placeholder="Ask about semiconductors..."></textarea>
        <button id="physbot-send" aria-label="Send">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
    `;
    document.body.appendChild(panel);

    // Render suggestions
    const suggContainer = document.getElementById('pb-suggestions');
    SUGGESTIONS.forEach(s => {
      const btn = document.createElement('button');
      btn.className = 'pb-sugg';
      btn.textContent = s;
      btn.addEventListener('click', () => sendMessage(s));
      suggContainer.appendChild(btn);
    });

    // Events
    fab.addEventListener('click', togglePanel);
    document.getElementById('pb-close-btn').addEventListener('click', togglePanel);
    document.getElementById('pb-clear-btn').addEventListener('click', clearChat);
    document.getElementById('physbot-send').addEventListener('click', onSend);
    const input = document.getElementById('physbot-input');
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); }
    });
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 100) + 'px';
    });

    // Welcome message
    setTimeout(() => addBotMessage(
      "👋 Hi! I'm **PhysBot**, your semiconductor physics AI assistant.\n\n" +
      "I retrieve live knowledge from **Wikipedia** and **ArXiv** on every query — " +
      "no stale manual knowledge base. I also connect to **llama3 via Ollama** if running locally.\n\n" +
      "Ask me anything about semiconductors, band theory, MOSFETs, Brillouin zones, " +
      "carrier transport, phonons, and more!",
      [], 0
    ), 300);
  }

  // ── Toggle ─────────────────────────────────────────────────
  function togglePanel() {
    isOpen = !isOpen;
    document.getElementById('physbot-panel').classList.toggle('open', isOpen);
    if (isOpen) {
      unread = 0;
      updateBadge();
      setTimeout(() => document.getElementById('physbot-input').focus(), 250);
    }
  }

  function updateBadge() {
    const badge = document.getElementById('pb-badge');
    badge.textContent = unread;
    badge.style.opacity = unread > 0 ? '1' : '0';
  }

  // ── Clear ──────────────────────────────────────────────────
  function clearChat() {
    history = [];
    document.getElementById('pb-messages').innerHTML = '';
    document.getElementById('pb-ctx-info').textContent = '';
    addBotMessage("Conversation cleared. Ready for new questions!", [], 0);
  }

  // ── Markdown renderer (no deps) ───────────────────────────
  function renderMarkdown(text) {
    return text
      // Code blocks
      .replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
        `<pre><code>${escHtml(code.trim())}</code></pre>`)
      // Inline code
      .replace(/`([^`]+)`/g, (_, c) => `<code>${escHtml(c)}</code>`)
      // Bold
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Italic
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Headers
      .replace(/^#{1,3}\s+(.+)$/gm, '<strong>$1</strong>')
      // Bullet lists
      .replace(/^[•\-\*]\s+(.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
      // Numbered lists
      .replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>')
      // Line breaks
      .replace(/\n\n+/g, '</p><p>')
      .replace(/\n/g, '<br>')
      .replace(/^/, '<p>').replace(/$/, '</p>');
  }

  function escHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  // ── Add messages ──────────────────────────────────────────
  function addUserMessage(text) {
    const container = document.getElementById('pb-messages');
    const div = document.createElement('div');
    div.className = 'pb-msg user';
    div.innerHTML = `
      <div class="pb-bubble">${escHtml(text)}</div>
      <div class="pb-mini-av">U</div>
    `;
    container.appendChild(div);
    scrollBottom();
  }

  function addBotMessage(text, sources=[], contextWords=0) {
    const container = document.getElementById('pb-messages');
    const div = document.createElement('div');
    div.className = 'pb-msg bot';

    const srcTags = sources.length
      ? sources.map(s => `<span class="pb-attr-tag">${escHtml(s)}</span>`).join('')
      : '';
    const ctxInfo = contextWords > 0
      ? `<span class="pb-ctx-count">${contextWords} ctx words</span>` : '';

    div.innerHTML = `
      <div class="pb-mini-av">⚛</div>
      <div class="pb-bubble">
        ${renderMarkdown(text)}
        ${(srcTags || ctxInfo) ? `<div class="pb-attribution">
          <span style="color:#2e4460;font-size:0.55rem;letter-spacing:0.1em;text-transform:uppercase;margin-right:4px">Sources:</span>
          ${srcTags}${ctxInfo}
        </div>` : ''}
      </div>
    `;
    container.appendChild(div);
    scrollBottom();

    if (!isOpen) { unread++; updateBadge(); }
  }

  function addTypingIndicator() {
    const container = document.getElementById('pb-messages');
    const div = document.createElement('div');
    div.className = 'pb-msg bot'; div.id = 'pb-typing-indicator';
    div.innerHTML = `
      <div class="pb-mini-av">⚛</div>
      <div class="pb-bubble">
        <div class="pb-typing">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    container.appendChild(div);
    scrollBottom();
  }

  function removeTypingIndicator() {
    const el = document.getElementById('pb-typing-indicator');
    if (el) el.remove();
  }

  function scrollBottom() {
    const c = document.getElementById('pb-messages');
    c.scrollTop = c.scrollHeight;
  }

  // ── Send ──────────────────────────────────────────────────
  function onSend() {
    const input = document.getElementById('physbot-input');
    const text  = input.value.trim();
    if (!text || isLoading) return;
    input.value = ''; input.style.height = 'auto';
    sendMessage(text);
  }

  function sendMessage(text) {
    if (!text || isLoading) return;
    isLoading = true;
    document.getElementById('pb-status-text').textContent = 'Retrieving...';
    document.getElementById('src-wiki').classList.remove('active');
    document.getElementById('src-arxiv').classList.remove('active');
    document.getElementById('src-ollama').classList.remove('active');

    addUserMessage(text);
    history.push({ role: 'user', content: text });
    addTypingIndicator();

    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, history: history.slice(-10) })
    })
    .then(r => r.json())
    .then(data => {
      removeTypingIndicator();
      const reply   = data.reply   || 'No response received.';
      const sources = data.sources || [];
      const engine  = data.engine  || '';
      const ctxWords = data.context_words || 0;

      history.push({ role: 'assistant', content: reply });
      addBotMessage(reply, sources, ctxWords);

      // Update source badges
      if (sources.some(s => s.toLowerCase().includes('wikipedia'))) {
        document.getElementById('src-wiki').classList.add('active');
      }
      if (sources.some(s => s.toLowerCase().includes('arxiv'))) {
        document.getElementById('src-arxiv').classList.add('active');
      }
      if (engine.includes('llama') || engine.includes('ollama')) {
        document.getElementById('src-ollama').classList.add('active');
      }
      if (ctxWords > 0) {
        document.getElementById('pb-ctx-info').textContent = `${ctxWords} words retrieved`;
      }
      document.getElementById('pb-status-text').textContent = 'Live RAG · Wikipedia + ArXiv';
    })
    .catch(err => {
      removeTypingIndicator();
      addBotMessage(`Connection error: ${err.message}. Is the Flask server running?`, [], 0);
      document.getElementById('pb-status-text').textContent = 'Error — check server';
    })
    .finally(() => { isLoading = false; });
  }

  // ── Init ──────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', buildUI);
  } else {
    buildUI();
  }

  // Warm up cache silently
  setTimeout(() => {
    fetch('/api/chat/warmup', { method: 'POST' }).catch(() => {});
  }, 2000);

})();
