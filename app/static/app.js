// ─── State ───
let chatMessages = [];
let sessions = [];
let currentSessionId = null;
let activeSessionId = null;

const LS_SESSIONS = 'rag_chat_sessions';
const LS_SETTINGS = 'rag_chat_settings';
const LS_ACTIVE_SESSION = 'rag_active_session_id';

const DEFAULT_SETTINGS = {
    top_k_retrieval: 5,
    top_k_rerank: 20,
    enable_reranker: false,
    reasoning: true,
    temperature: 0.0,
    top_k: 20,
    top_p: 1.0,
    min_p: 0.0,
    enable_prompt_guardrail: false,
    enable_stream_loop_guardrail: false,
    enable_grounding_guardrail: false,
    enable_language_guardrail: false
};

let settings = {};
let modelNames = { reasoning: 'gemma-4-e2b', nonReasoning: 'gemma-4-e2b-instruct' };

// ─── DOM Elements ───
const welcomeScreen = document.getElementById('welcomeScreen');
const chatHistory = document.getElementById('chatHistory');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const newSessionBtn = document.getElementById('newSessionBtn');
const sessionList = document.getElementById('sessionList');
const suggestionsContainer = document.getElementById('suggestionsContainer');
const exampleChips = document.querySelectorAll('.configurator-option-chip');

// Settings inputs
const inputTopKRetrieval = document.getElementById('inputTopKRetrieval');
const toggleReranker = document.getElementById('toggleReranker');
const inputTopKRerank = document.getElementById('inputTopKRerank');
const rerankPoolRow = document.getElementById('rerankPoolRow');
const inputTemperature = document.getElementById('inputTemperature');
const displayTemperature = document.getElementById('displayTemperature');
const inputTopKSampling = document.getElementById('inputTopKSampling');
const inputTopP = document.getElementById('inputTopP');
const displayTopP = document.getElementById('displayTopP');
const inputMinP = document.getElementById('inputMinP');
const displayMinP = document.getElementById('displayMinP');
const togglePromptGuardrail = document.getElementById('togglePromptGuardrail');
const toggleStreamLoopGuardrail = document.getElementById('toggleStreamLoopGuardrail');
const toggleGroundingGuardrail = document.getElementById('toggleGroundingGuardrail');
const togglePreventCjk = document.getElementById('togglePreventCjk');
const toggleReasoning = document.getElementById('toggleReasoning');

// Modal
const citationModal = document.getElementById('citationModal');
const modalTitle = document.getElementById('modalTitle');
const modalDocTitle = document.getElementById('modalDocTitle');
const modalDocNumber = document.getElementById('modalDocNumber');
const modalContent = document.getElementById('modalContent');
const modalClose = document.getElementById('modalClose');

// ─── Init ───
document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    applySettingsToUI();
    bindSettingsUI();
    bindToggleHandlers();
    loadSessions();
    restoreActiveSession();
    setupEventListeners();
    bindSessionListClick();
    updateRerankPoolVisibility();
    fetchModelName();
    if (typeof lucide !== 'undefined' && lucide.createIcons) {
        lucide.createIcons();
    }
});

function updateModelNameDisplay() {
    const modelNameEl = document.querySelector('.model-name');
    if (!modelNameEl) return;
    let modelName = settings.reasoning ? modelNames.reasoning : modelNames.nonReasoning;
    const maxLen = 18;
    if (modelName && modelName.length > maxLen) {
        modelName = modelName.substring(0, maxLen - 3) + '...';
    }
    modelNameEl.textContent = modelName || 'SMEBot';
}

async function fetchModelName() {
    try {
        const resp = await fetch('/api/status');
        if (resp.ok) {
            const data = await resp.json();
            if (data.reasoning_model) {
                modelNames.reasoning = data.reasoning_model;
            }
            if (data.non_reasoning_model) {
                modelNames.nonReasoning = data.non_reasoning_model;
            }
            updateModelNameDisplay();
        }
    } catch (e) {
        console.error("Failed to fetch model status", e);
    }
}

// ─── Settings ───
function loadSettings() {
    try {
        const raw = localStorage.getItem(LS_SETTINGS);
        settings = raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : { ...DEFAULT_SETTINGS };
    } catch {
        settings = { ...DEFAULT_SETTINGS };
    }
}

function saveSettings() {
    localStorage.setItem(LS_SETTINGS, JSON.stringify(settings));
}

function applySettingsToUI() {
    inputTopKRetrieval.value = settings.top_k_retrieval;
    setToggle(toggleReranker, settings.enable_reranker);
    inputTopKRerank.value = settings.top_k_rerank;
    inputTemperature.value = settings.temperature;
    displayTemperature.textContent = settings.temperature.toFixed(1);
    inputTopKSampling.value = settings.top_k;
    inputTopP.value = settings.top_p;
    displayTopP.textContent = settings.top_p.toFixed(2);
    inputMinP.value = settings.min_p;
    displayMinP.textContent = settings.min_p.toFixed(2);
    setToggle(togglePromptGuardrail, settings.enable_prompt_guardrail);
    setToggle(toggleStreamLoopGuardrail, settings.enable_stream_loop_guardrail);
    setToggle(toggleGroundingGuardrail, settings.enable_grounding_guardrail);
    setToggle(togglePreventCjk, settings.enable_language_guardrail);
    setToggle(toggleReasoning, settings.reasoning);
    updateModelNameDisplay();
}

function readSettingsFromUI() {
    settings.top_k_retrieval = parseInt(inputTopKRetrieval.value) || 5;
    settings.top_k_rerank = parseInt(inputTopKRerank.value) || 20;
    settings.enable_reranker = toggleReranker.classList.contains('active');
    settings.temperature = parseFloat(inputTemperature.value) || 0;
    settings.top_k = parseInt(inputTopKSampling.value) || 20;
    settings.top_p = parseFloat(inputTopP.value) || 1.0;
    settings.min_p = parseFloat(inputMinP.value) || 0;
    settings.enable_prompt_guardrail = togglePromptGuardrail.classList.contains('active');
    settings.enable_stream_loop_guardrail = toggleStreamLoopGuardrail.classList.contains('active');
    settings.enable_grounding_guardrail = toggleGroundingGuardrail.classList.contains('active');
    settings.enable_language_guardrail = togglePreventCjk.classList.contains('active');
    settings.reasoning = toggleReasoning.classList.contains('active');
}

function bindSettingsUI() {
    // Slider displays
    inputTemperature.addEventListener('input', () => {
        displayTemperature.textContent = parseFloat(inputTemperature.value).toFixed(1);
        readSettingsFromUI(); saveSettings();
    });
    inputTopP.addEventListener('input', () => {
        displayTopP.textContent = parseFloat(inputTopP.value).toFixed(2);
        readSettingsFromUI(); saveSettings();
    });
    inputMinP.addEventListener('input', () => {
        displayMinP.textContent = parseFloat(inputMinP.value).toFixed(2);
        readSettingsFromUI(); saveSettings();
    });
    // Number inputs
    [inputTopKRetrieval, inputTopKRerank, inputTopKSampling].forEach(el => {
        el.addEventListener('change', () => { readSettingsFromUI(); saveSettings(); });
    });
}

function setToggle(el, on) {
    el.classList.toggle('active', on);
    el.setAttribute('aria-checked', on.toString());
}

function bindToggleHandlers() {
    const toggles = [toggleReranker, togglePromptGuardrail, toggleStreamLoopGuardrail, toggleGroundingGuardrail, togglePreventCjk, toggleReasoning];
    toggles.forEach(el => {
        el.addEventListener('click', () => {
            const isOn = el.classList.toggle('active');
            el.setAttribute('aria-checked', isOn.toString());
            readSettingsFromUI(); saveSettings();
            if (el === toggleReranker) updateRerankPoolVisibility();
            if (el === toggleReasoning) updateModelNameDisplay();
        });
    });
}

function updateRerankPoolVisibility() {
    rerankPoolRow.style.display = toggleReranker.classList.contains('active') ? 'flex' : 'none';
}

// ─── Settings collapsible ───
document.querySelectorAll('.settings-section-header').forEach(hdr => {
    hdr.addEventListener('click', () => {
        hdr.classList.toggle('collapsed');
    });
});

// ─── Sessions ───
function loadSessions() {
    try {
        const raw = localStorage.getItem(LS_SESSIONS);
        sessions = raw ? JSON.parse(raw) : [];
    } catch {
        sessions = [];
    }
}

function saveSessions() {
    localStorage.setItem(LS_SESSIONS, JSON.stringify(sessions));
}

function getActiveSession() {
    return sessions.find(s => s.id === activeSessionId) || null;
}

function createNewSession() {
    const id = 'session_' + Date.now();
    const session = { id, title: '', timestamp: Date.now(), messages: [] };
    sessions.unshift(session);
    activeSessionId = id;
    saveActiveSessionId();
    saveSessions();
    renderSessionList();
    return session;
}

function restoreActiveSession() {
    activeSessionId = localStorage.getItem(LS_ACTIVE_SESSION);
    const session = getActiveSession();
    if (session && session.messages.length > 0) {
        renderSessionMessages(session);
        hideWelcomeElements();
    } else {
        showWelcomeElements();
    }
    renderSessionList();
    highlightActiveSession();
}

function saveActiveSessionId() {
    if (activeSessionId) localStorage.setItem(LS_ACTIVE_SESSION, activeSessionId);
}

function switchToSession(id) {
    activeSessionId = id;
    saveActiveSessionId();
    chatHistory.innerHTML = '';
    chatMessages = [];
    const session = getActiveSession();
    if (session && session.messages.length > 0) {
        renderSessionMessages(session);
        hideWelcomeElements();
    } else {
        showWelcomeElements();
    }
    renderSessionList();
    highlightActiveSession();
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function renderSessionMessages(session) {
    for (const msg of session.messages) {
        if (msg.role === 'user') {
            appendMessageBubble('user', msg.content);
        } else if (msg.role === 'assistant') {
            appendMessageBubble('assistant', msg.content, msg.chunks || [], msg.citations || [], true, msg.reasoning);
        }
    }
}

function renderSessionList() {
    sessionList.innerHTML = '';
    for (const s of sessions) {
        const div = document.createElement('div');
        div.className = 'session-item' + (s.id === activeSessionId ? ' active' : '');
        div.dataset.sessionId = s.id;
        const title = s.title || '(Cuộc hội thoại mới)';
        const time = new Date(s.timestamp).toLocaleString('vi-VN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        div.innerHTML = `<div class="session-item-title">${escapeHtml(title)}</div><div class="session-item-time">${time}</div>`;
        sessionList.appendChild(div);
    }
}

function highlightActiveSession() {
    sessionList.querySelectorAll('.session-item').forEach(el => {
        el.classList.toggle('active', el.dataset.sessionId === activeSessionId);
    });
}

function bindSessionListClick() {
    sessionList.addEventListener('click', e => {
        const item = e.target.closest('.session-item');
        if (item) switchToSession(item.dataset.sessionId);
    });
}

function addMessageToSession(role, content, chunks, citations, reasoning) {
    let session = getActiveSession();
    if (!session) {
        session = createNewSession();
    }
    if (role === 'user' && !session.title) {
        session.title = content.substring(0, 80);
        session.timestamp = Date.now();
    }
    session.messages.push({ role, content, chunks: chunks || [], citations: citations || [], reasoning: reasoning || '', timestamp: Date.now() });
    session.timestamp = Date.now();
    saveSessions();
    renderSessionList();
    highlightActiveSession();
}

// ─── Event Listeners ───
function setupEventListeners() {
    sendBtn.addEventListener('click', submitMessage);
    chatInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.isComposing && !e.shiftKey) {
            e.preventDefault();
            submitMessage();
        }
    });
    clearHistoryBtn.addEventListener('click', clearAllHistory);
    newSessionBtn.addEventListener('click', showHomeScreen);
    exampleChips.forEach(chip => {
        chip.addEventListener('click', () => {
            chatInput.value = chip.getAttribute('data-query');
            submitMessage();
        });
    });
    modalClose.addEventListener('click', closeCitation);
    citationModal.addEventListener('click', e => { if (e.target === citationModal) closeCitation(); });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && citationModal.classList.contains('active')) closeCitation();
    });

    // ─── Collapsible Sidebar Right ───
    const sidebarRight = document.getElementById('sidebarRight');
    const toggleRightSidebarBtn = document.getElementById('toggleRightSidebarBtn');
    const closeRightSidebarBtn = document.getElementById('closeRightSidebarBtn');

    if (toggleRightSidebarBtn && sidebarRight) {
        toggleRightSidebarBtn.addEventListener('click', (e) => {
            e.preventDefault();
            sidebarRight.classList.toggle('collapsed');
        });
    }
    if (closeRightSidebarBtn && sidebarRight) {
        closeRightSidebarBtn.addEventListener('click', (e) => {
            e.preventDefault();
            sidebarRight.classList.add('collapsed');
        });
    }

    // ─── Left Sidebar Mock Menu Handlers ───
    ['menuHome', 'menuExplore', 'menuLibrary', 'menuHistory'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('click', (e) => {
                e.preventDefault();
                document.querySelectorAll('.menu-item').forEach(m => m.classList.remove('active'));
                el.classList.add('active');
                if (id === 'menuHome') {
                    showHomeScreen();
                }
            });
        }
    });

    // ─── Chat Input Auto-grow Height ───
    if (chatInput) {
        const updateHeight = () => {
            // Adjust height
            chatInput.style.height = 'auto';
            chatInput.style.height = (chatInput.scrollHeight - 12) + 'px';
        };
        chatInput.addEventListener('input', updateHeight);
    }
}

// ─── Submit ───
async function submitMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    hideWelcomeElements();
    chatInput.value = '';
    chatInput.style.height = 'auto'; // Reset height

    appendMessageBubble('user', query);
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

    addMessageToSession('user', query);

    const loaderId = appendTypingIndicator();
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

    readSettingsFromUI();
    setInputEnabled(false);

    try {
        const resp = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                reasoning: settings.reasoning,
                top_k_retrieval: settings.top_k_retrieval,
                top_k_rerank: settings.top_k_rerank,
                temperature: settings.temperature,
                top_k: settings.top_k,
                top_p: settings.top_p,
                min_p: settings.min_p,
                enable_reranker: settings.enable_reranker,
                enable_prompt_guardrail: settings.enable_prompt_guardrail,
                enable_stream_loop_guardrail: settings.enable_stream_loop_guardrail,
                enable_grounding_guardrail: settings.enable_grounding_guardrail,
                enable_language_guardrail: settings.enable_language_guardrail
            })
        });

        if (!resp.ok) {
            removeTypingIndicator(loaderId);
            appendMessageBubble('assistant', 'Không thể hoàn thành yêu cầu do lỗi máy chủ.');
            return;
        }

        removeTypingIndicator(loaderId);
        const streaming = createStreamingAssistantBubble();
        const reader = resp.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buf = '';
        let loading = true;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const parts = buf.split('\n\n');
            buf = parts.pop();
            for (const part of parts) {
                const line = part.trim();
                if (!line) continue;
                if (line.startsWith('data:')) {
                    let ev;
                    try { ev = JSON.parse(line.replace(/^data:\s*/, '')); } catch { continue; }
                    if (ev.type === 'chunk') {
                        if (loading) { removeTypingIndicator(loaderId); loading = false; }
                        updateStreamingAssistantBubble(streaming, ev.content || ev.text || '');
                    } else if (ev.type === 'reasoning') {
                        if (loading) { removeTypingIndicator(loaderId); loading = false; }
                        updateStreamingReasoningOnly(streaming, ev.content || '');
                    } else if (ev.type === 'error' || ev.type === 'invalidate') {
                        removeTypingIndicator(loaderId);
                        let errMsg = ev.answer || 'Không thể hoàn thành yêu cầu.';
                        if (ev.message === 'stream_loop_detected') errMsg = 'Phát hiện vòng lặp token. Vui lòng thử lại.';
                        invalidateStreamingAssistantBubble(streaming, errMsg);
                        return;
                    } else if (ev.type === 'done') {
                        removeTypingIndicator(loaderId);
                        finalizeStreamingAssistantBubble(streaming, ev.answer || '', ev.chunks || [], ev.citations || [], ev.is_valid);
                        addMessageToSession('assistant', ev.answer || '', ev.chunks || [], ev.citations || [], streaming.currentReasoning);
                        return;
                    }
                }
            }
        }
        removeTypingIndicator(loaderId);
        const finalText = streaming.currentTextFull || '';
        if (finalText) {
            finalizeStreamingAssistantBubble(streaming, finalText, [], [], false);
            addMessageToSession('assistant', finalText, [], []);
        }
    } catch (error) {
        removeTypingIndicator(loaderId);
        appendMessageBubble('assistant', 'Lỗi kết nối mạng đến máy chủ backend.');
    } finally {
        setInputEnabled(true);
    }
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function showHomeScreen() {
    chatMessages = [];
    chatHistory.innerHTML = '';
    if (chatInput) {
        chatInput.value = '';
        chatInput.style.height = 'auto';
    }

    activeSessionId = null;
    localStorage.removeItem(LS_ACTIVE_SESSION);
    showWelcomeElements();
    renderSessionList();
    highlightActiveSession();
    
    // Also highlight the Home menu item in left sidebar
    document.querySelectorAll('.menu-item').forEach(m => m.classList.remove('active'));
    const menuHome = document.getElementById('menuHome');
    if (menuHome) menuHome.classList.add('active');
    
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function clearAllHistory() {
    if (!confirm('Xoá toàn bộ lịch sử hội thoại?')) return;
    sessions = [];
    activeSessionId = null;
    localStorage.removeItem(LS_SESSIONS);
    localStorage.removeItem(LS_ACTIVE_SESSION);
    chatMessages = [];
    chatHistory.innerHTML = '';
    showWelcomeElements();
    renderSessionList();
}

// ─── Transitions ───
function setInputEnabled(enabled) {
    chatInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
    chatInput.classList.toggle('input-disabled', !enabled);
    sendBtn.classList.toggle('send-disabled', !enabled);
}



let activeTypewriterTimeouts = [];

function clearAllTypewriters() {
    activeTypewriterTimeouts.forEach(id => clearTimeout(id));
    activeTypewriterTimeouts = [];
}

function prepareTypeWriter(element, text) {
    if (!element) return [];
    element.innerHTML = '';

    const words = text.split(' ');
    const spansToReveal = [];

    words.forEach((word, wordIdx) => {
        const wordSpan = document.createElement('span');
        wordSpan.className = 'typing-word';

        const chars = Array.from(word);
        chars.forEach(char => {
            const charSpan = document.createElement('span');
            charSpan.textContent = char;
            charSpan.className = 'typing-char';
            wordSpan.appendChild(charSpan);
            spansToReveal.push(charSpan);
        });

        element.appendChild(wordSpan);

        if (wordIdx < words.length - 1) {
            const spaceSpan = document.createElement('span');
            spaceSpan.textContent = ' ';
            spaceSpan.className = 'typing-char';
            element.appendChild(spaceSpan);
            spansToReveal.push(spaceSpan);
        }
    });

    return spansToReveal;
}

function revealSpans(spans, speed = 15, callback) {
    let i = 0;
    function reveal() {
        if (i < spans.length) {
            spans[i].classList.add('revealed');
            i++;
            const timeoutId = setTimeout(reveal, speed);
            activeTypewriterTimeouts.push(timeoutId);
        } else if (callback) {
            callback();
        }
    }
    reveal();
}

function showRandomGreeting() {
    clearAllTypewriters();
    const titleEl = document.getElementById('welcomeTitle');
    const subtitleEl = document.getElementById('welcomeSubtitle');
    if (!titleEl || !subtitleEl) return;

    const hour = new Date().getHours();
    let greeting;
    if (hour >= 5 && hour < 12) {
        greeting = {
            title: "Chào buổi sáng!",
            subtitle: "Chúc bạn một ngày mới tốt lành. Tôi có thể giúp gì cho bạn hôm nay?"
        };
    } else if (hour >= 12 && hour < 18) {
        greeting = {
            title: "Chào buổi chiều!",
            subtitle: "Tôi có thể giúp gì cho bạn lúc này?"
        };
    } else {
        greeting = {
            title: "Chào buổi tối!",
            subtitle: "Tôi có thể giúp gì cho bạn tối nay?"
        };
    }

    const titleSpans = prepareTypeWriter(titleEl, greeting.title);
    const subtitleSpans = prepareTypeWriter(subtitleEl, greeting.subtitle);

    revealSpans(titleSpans, 15, () => {
        revealSpans(subtitleSpans, 15);
    });
}

function hideWelcomeElements() {
    if (welcomeScreen && welcomeScreen.style.display !== 'none') {
        welcomeScreen.style.transition = 'opacity 0.25s ease-out';
        welcomeScreen.style.opacity = '0';
        setTimeout(() => { welcomeScreen.style.display = 'none'; }, 250);
    }
}

function showWelcomeElements() {
    if (welcomeScreen) {
        welcomeScreen.style.display = 'flex';
        welcomeScreen.style.opacity = '1';
        showRandomGreeting();
    }
}

// ─── UI Rendering ───
// ─── UI Rendering ───
function appendMessageBubble(role, content, chunks = [], citations = [], isValid = true, reasoning = '') {
    const msgIndex = chatMessages.length;
    chatMessages.push({ role, content, chunks, citations });

    const rowDiv = document.createElement('div');
    rowDiv.className = `chat-message-row ${role}`;
    rowDiv.setAttribute('data-msg-idx', msgIndex);

    // Avatar container
    const avatarDiv = document.createElement('div');
    avatarDiv.className = `message-avatar ${role}-avatar`;
    if (role === 'user') {
        avatarDiv.innerHTML = `<i data-lucide="user"></i>`;
    } else {
        avatarDiv.innerHTML = `<img src="/static/logo.svg?v=20" alt="SMEBot Logo">`;
    }

    // Content container
    const contentContainer = document.createElement('div');
    contentContainer.className = 'message-content-container';

    if (role === 'assistant') {
        const nameHeader = document.createElement('div');
        nameHeader.className = 'sender-name';
        nameHeader.textContent = 'SMEBot';
        contentContainer.appendChild(nameHeader);
    }

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'chat-bubble';

    if (content === 'Không thể trả lời câu hỏi này') {
        bubbleDiv.className += ' insufficient-info';
        bubbleDiv.textContent = content;
    } else if (role === 'user') {
        bubbleDiv.textContent = content;
    } else {
        if (reasoning) {
            const thinkRendered = renderMarkdown(reasoning);
            const thinkHtml = `<div class="think-container">
  <button class="think-toggle">
    <span class="think-text">Xem suy luận</span>
    <svg class="think-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
  </button>
  <div class="think-body hidden">
    <div class="think-body-content">${thinkRendered}</div>
  </div>
</div>`;
            bubbleDiv.insertAdjacentHTML('beforeend', thinkHtml);
        }
        let processedText = renderMarkdown(content);
        processedText = processedText.replace(/\[(\d+)\]/g, (match, citId) => {
            const id = parseInt(citId);
            if (id < chunks.length) {
                const chunk = chunks[id];
                const tooltip = `${chunk.doc_title} - ${chunk.article_number}\n${chunk.article_content.substring(0, 100).trim()}...`;
                const tooltipEsc = tooltip.replace(/"/g, '&quot;').replace(/'/g, '&apos;');
                return `<span class="cit-badge" data-cit-id="${id}" title="${tooltipEsc}">${id}</span>`;
            }
            return match;
        });
        if (reasoning) {
            const contentDiv = document.createElement('div');
            contentDiv.className = 'assistant-text-content';
            contentDiv.innerHTML = processedText;
            bubbleDiv.appendChild(contentDiv);
        } else {
            bubbleDiv.innerHTML = processedText;
        }
    }

    contentContainer.appendChild(bubbleDiv);

    if (role === 'assistant' && citations.length > 0 && chunks.length > 0 && content !== 'Không thể trả lời câu hỏi này') {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        const label = document.createElement('span');
        label.className = 'source-label';
        label.textContent = 'Nguồn trích dẫn:';
        sourcesDiv.appendChild(label);
        const uniqueCids = [...new Set(citations)].sort((a, b) => a - b);
        uniqueCids.forEach(cid => {
            if (cid < chunks.length) {
                const chunk = chunks[cid];
                const pill = document.createElement('span');
                pill.className = 'source-pill';
                pill.setAttribute('data-cit-id', cid);
                pill.innerHTML = `Xem ${chunk.article_number} - ${chunk.doc_number}`;
                sourcesDiv.appendChild(pill);
            }
        });
        contentContainer.appendChild(sourcesDiv);
    }

    rowDiv.appendChild(avatarDiv);
    rowDiv.appendChild(contentContainer);
    chatHistory.appendChild(rowDiv);

    if (typeof lucide !== 'undefined' && lucide.createIcons) {
        lucide.createIcons();
    }

    bindCitationClickHandlers(rowDiv, msgIndex);
    if (role === 'assistant' && reasoning) {
        attachThinkToggleHandlers(bubbleDiv);
    }
}

function appendTypingIndicator() {
    const id = 'loader-' + Date.now();
    const rowDiv = document.createElement('div');
    rowDiv.className = 'chat-message-row assistant';
    rowDiv.id = id;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar assistant-avatar';
    avatarDiv.innerHTML = `<img src="/static/logo.svg?v=20" alt="SMEBot Logo">`;

    const contentContainer = document.createElement('div');
    contentContainer.className = 'message-content-container';

    const nameHeader = document.createElement('div');
    nameHeader.className = 'sender-name';
    nameHeader.textContent = 'SMEBot';
    contentContainer.appendChild(nameHeader);

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'chat-bubble';
    bubbleDiv.innerHTML = `<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
    contentContainer.appendChild(bubbleDiv);

    rowDiv.appendChild(avatarDiv);
    rowDiv.appendChild(contentContainer);
    chatHistory.appendChild(rowDiv);
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function createStreamingAssistantBubble() {
    const msgIndex = chatMessages.length;
    const rowDiv = document.createElement('div');
    rowDiv.className = 'chat-message-row assistant';
    rowDiv.setAttribute('data-msg-idx', msgIndex);

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar assistant-avatar';
    avatarDiv.innerHTML = `<img src="/static/logo.svg?v=20" alt="SMEBot Logo">`;

    const contentContainer = document.createElement('div');
    contentContainer.className = 'message-content-container';

    const nameHeader = document.createElement('div');
    nameHeader.className = 'sender-name';
    nameHeader.textContent = 'SMEBot';
    contentContainer.appendChild(nameHeader);

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'chat-bubble';
    bubbleDiv.innerHTML = `<div class="think-container">
  <button class="think-toggle disabled shimmering">
    <span class="think-text">Đang chuẩn bị</span>
  </button>
  <div class="think-body hidden">
    <div class="think-body-content"></div>
  </div>
</div>`;
    contentContainer.appendChild(bubbleDiv);

    rowDiv.appendChild(avatarDiv);
    rowDiv.appendChild(contentContainer);
    chatHistory.appendChild(rowDiv);
    attachThinkToggleHandlers(bubbleDiv);

    chatMessages.push({ role: 'assistant', content: '', chunks: [], citations: [] });
    return { msgIndex, messageDiv: rowDiv, bubbleDiv, currentTextFull: '', currentReasoning: '' };
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderMarkdown(text) {
    if (!text) return '';
    const escaped = escapeHtml(text);
    return marked.parse(escaped, { breaks: true });
}

function updateStreamingAssistantBubble(streamingObj, tokenText) {
    if (!streamingObj) return;
    const wasEmpty = streamingObj.currentTextFull === '';
    streamingObj.currentTextFull += tokenText;
    if (wasEmpty) {
        if (streamingObj.currentReasoning === '') {
            const thinkContainer = streamingObj.bubbleDiv.querySelector('.think-container');
            if (thinkContainer) thinkContainer.remove();
        }
    }
    const rendered = renderMarkdown(streamingObj.currentTextFull);
    let answerDiv = streamingObj.bubbleDiv.querySelector('.answer-content');
    if (answerDiv) {
        answerDiv.innerHTML = rendered;
    } else {
        answerDiv = document.createElement('div');
        answerDiv.className = 'answer-content';
        answerDiv.innerHTML = rendered;
        const thinkContainer = streamingObj.bubbleDiv.querySelector('.think-container');
        if (thinkContainer) thinkContainer.after(answerDiv);
        else streamingObj.bubbleDiv.appendChild(answerDiv);
    }
    chatMessages[streamingObj.msgIndex].content = streamingObj.currentTextFull;
}

function updateStreamingReasoningOnly(streamingObj, reasoningText) {
    if (!streamingObj) return;
    const wasEmpty = streamingObj.currentReasoning === '';
    streamingObj.currentReasoning += reasoningText;
    if (wasEmpty) {
        const preparing = streamingObj.bubbleDiv.querySelector('.preparing-text');
        if (preparing) preparing.remove();
    }
    let thinkContainer = streamingObj.bubbleDiv.querySelector('.think-container');
    if (!thinkContainer) {
        const rendered = renderMarkdown(streamingObj.currentReasoning);
        const thinkHtml = `<div class="think-container">
  <button class="think-toggle">
    <span class="think-text">Đang suy nghĩ...</span>
    <svg class="think-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
  </button>
  <div class="think-body hidden">
    <div class="think-body-content">${rendered}</div>
  </div>
</div>`;
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = thinkHtml;
        thinkContainer = tempDiv.firstChild;
        streamingObj.bubbleDiv.insertBefore(thinkContainer, streamingObj.bubbleDiv.firstChild);
        attachThinkToggleHandlers(streamingObj.bubbleDiv);
    }

    if (wasEmpty) {
        const toggle = thinkContainer.querySelector('.think-toggle');
        const textEl = thinkContainer.querySelector('.think-text');
        if (toggle) {
            toggle.classList.remove('disabled');
            toggle.classList.remove('shimmering');
            if (!toggle.querySelector('.think-chevron')) {
                const chevronSvg = `<svg class="think-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>`;
                toggle.insertAdjacentHTML('beforeend', chevronSvg);
            }
        }
        if (textEl) {
            textEl.textContent = 'Đang suy nghĩ...';
        }
    }

    const thinkBodyContent = thinkContainer.querySelector('.think-body-content');
    const thinkBody = thinkContainer.querySelector('.think-body');
    if (thinkBodyContent && thinkBody) {
        const wasAtBottom = thinkBody.scrollHeight - thinkBody.scrollTop - thinkBody.clientHeight < 30;
        thinkBodyContent.innerHTML = renderMarkdown(streamingObj.currentReasoning);
        if (wasAtBottom && !thinkBody.classList.contains('user-scrolled')) {
            thinkBody.scrollTop = thinkBody.scrollHeight;
        }
    }
    chatMessages[streamingObj.msgIndex].content = streamingObj.currentTextFull + '\n<think>' + streamingObj.currentReasoning + '</think>';
}

function renderAssistantContentWithCitations(content, chunks) {
    let processedText = renderMarkdown(content);
    processedText = processedText.replace(/\[(\d+)\]/g, (match, citId) => {
        const id = parseInt(citId);
        if (id < chunks.length) {
            const chunk = chunks[id];
            const tooltip = `${chunk.doc_title} - ${chunk.article_number}\n${chunk.article_content.substring(0, 100).trim()}...`;
            const tooltipEsc = tooltip.replace(/"/g, '&quot;').replace(/'/g, '&apos;');
            return `<span class="cit-badge" data-cit-id="${id}" title="${tooltipEsc}">${id}</span>`;
        }
        return match;
    });
    return processedText;
}

function finalizeStreamingAssistantBubble(streamingObj, finalText, chunks, citations, isValid) {
    if (!streamingObj) return;
    const finalHtml = renderAssistantContentWithCitations(finalText, chunks);
    streamingObj.bubbleDiv.innerHTML = '';
    if (streamingObj.currentReasoning) {
        const rendered = renderMarkdown(streamingObj.currentReasoning);
        const thinkHtml = `<div class="think-container">
  <button class="think-toggle">
    <span class="think-text">Xem suy luận</span>
    <svg class="think-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
  </button>
  <div class="think-body hidden">
    <div class="think-body-content">${rendered}</div>
  </div>
</div>`;
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = thinkHtml;
        streamingObj.bubbleDiv.appendChild(tempDiv.firstChild);
    }
    const contentDiv = document.createElement('div');
    contentDiv.className = 'assistant-text-content';
    contentDiv.innerHTML = finalHtml;
    streamingObj.bubbleDiv.appendChild(contentDiv);
    attachThinkToggleHandlers(streamingObj.bubbleDiv);
    chatMessages[streamingObj.msgIndex].content = finalText;
    chatMessages[streamingObj.msgIndex].chunks = chunks;
    chatMessages[streamingObj.msgIndex].citations = citations;
    if (citations.length > 0 && chunks.length > 0 && finalText !== 'Không thể trả lời câu hỏi này') {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        const label = document.createElement('span');
        label.className = 'source-label';
        label.textContent = 'Nguồn trích dẫn:';
        sourcesDiv.appendChild(label);
        const uniqueCids = [...new Set(citations)].sort((a, b) => a - b);
        uniqueCids.forEach(cid => {
            if (cid < chunks.length) {
                const chunk = chunks[cid];
                const pill = document.createElement('span');
                pill.className = 'source-pill';
                pill.setAttribute('data-cit-id', cid);
                pill.innerHTML = `Xem ${chunk.article_number} - ${chunk.doc_number}`;
                sourcesDiv.appendChild(pill);
            }
        });
        streamingObj.messageDiv.querySelector('.message-content-container').appendChild(sourcesDiv);
    }
    bindCitationClickHandlers(streamingObj.messageDiv, streamingObj.msgIndex);
}

function invalidateStreamingAssistantBubble(streamingObj, replacementText) {
    if (!streamingObj) return;
    streamingObj.bubbleDiv.innerHTML = replacementText;
    chatMessages[streamingObj.msgIndex].content = replacementText;
    chatMessages[streamingObj.msgIndex].chunks = [];
    chatMessages[streamingObj.msgIndex].citations = [];
}

// ─── Citation Handlers ───
function bindCitationClickHandlers(parentEl, msgIndex) {
    const badges = parentEl.querySelectorAll('.cit-badge');
    const pills = parentEl.querySelectorAll('.source-pill');
    const handler = e => {
        const citId = parseInt(e.currentTarget.getAttribute('data-cit-id'));
        openCitation(msgIndex, citId, e);
    };
    badges.forEach(b => b.addEventListener('click', handler));
    pills.forEach(p => p.addEventListener('click', handler));
}

function openCitation(msgIndex, citId, mouseEvent) {
    const message = chatMessages[msgIndex];
    if (!message || !message.chunks || citId >= message.chunks.length) return;
    const chunk = message.chunks[citId];
    modalTitle.textContent = `Chi tiết trích dẫn [${citId}]: ${chunk.article_number}`;
    modalDocTitle.textContent = chunk.doc_title;
    modalDocNumber.textContent = chunk.doc_number;
    modalContent.textContent = chunk.article_content;

    const container = citationModal.querySelector('.citation-modal-container');
    if (container && mouseEvent) {
        let x = mouseEvent.clientX + 10;
        let y = mouseEvent.clientY + 10;

        const popoverWidth = 360;
        if (x + popoverWidth > window.innerWidth) {
            x = window.innerWidth - popoverWidth - 20;
        }

        const popoverHeight = 300;
        if (y + popoverHeight > window.innerHeight) {
            y = window.innerHeight - popoverHeight - 20;
        }

        if (x < 10) x = 10;
        if (y < 10) y = 10;

        container.style.position = 'fixed';
        container.style.left = `${x}px`;
        container.style.top = `${y}px`;
        container.style.margin = '0';
    }

    citationModal.classList.add('active');
}

function closeCitation() {
    citationModal.classList.remove('active');
}

// ─── Think Toggle ───
function attachThinkToggleHandlers(bubbleDiv) {
    const toggle = bubbleDiv.querySelector('.think-toggle');
    if (!toggle) return;
    const body = bubbleDiv.querySelector('.think-body');
    if (!body) return;
    const newToggle = toggle.cloneNode(true);
    toggle.parentNode.replaceChild(newToggle, toggle);
    newToggle.addEventListener('click', e => {
        e.preventDefault();
        const isHidden = body.classList.toggle('hidden');
        newToggle.classList.toggle('expanded', !isHidden);
        if (!isHidden) body.scrollTop = body.scrollHeight;
    });
    body.addEventListener('scroll', () => {
        const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 30;
        body.classList.toggle('user-scrolled', !atBottom);
    }, { passive: true });
}