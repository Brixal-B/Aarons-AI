// Chat application JavaScript

const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const ragPanel = document.getElementById('rag-panel');
const folderPathInput = document.getElementById('folder-path');
const loadBtn = document.getElementById('load-btn');
const ragStatusDot = document.getElementById('rag-status-dot');
const ragStatusText = document.getElementById('rag-status-text');
const ragEnabledCheckbox = document.getElementById('rag-enabled');
const modelSelect = document.getElementById('model-select');
const sidebar = document.getElementById('sidebar');
const conversationList = document.getElementById('conversation-list');
const conversationTitle = document.getElementById('conversation-title');

let isGenerating = false;
let currentModel = modelSelect ? modelSelect.value : 'llama3.2';
let sessionId = crypto.randomUUID();
let ragLoaded = false;
let ragEnabled = false;
let currentConversationId = null;
let currentConversationName = 'New Chat';
let conversationMessages = []; // Track messages for saving
let allConversations = []; // Cache of conversation list

// Configure marked.js
marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return hljs.highlight(code, { language: lang }).value;
            } catch (e) {}
        }
        return hljs.highlightAuto(code).value;
    }
});

// SVG icons
const copyIcon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
const checkIcon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function toggleRagPanel() {
    ragPanel.classList.toggle('collapsed');
}

function updateRagMode() {
    ragEnabled = ragEnabledCheckbox.checked && ragLoaded;
    if (ragEnabledCheckbox.checked && !ragLoaded) {
        ragEnabledCheckbox.checked = false;
        alert('Please load documents first before enabling RAG mode.');
    }
}

async function loadDocuments() {
    const folderPath = folderPathInput.value.trim();
    if (!folderPath) {
        alert('Please enter a folder path.');
        return;
    }

    loadBtn.disabled = true;
    loadBtn.textContent = 'Loading...';
    ragStatusDot.className = 'rag-status-dot loading';
    ragStatusText.textContent = 'Loading documents...';

    try {
        const response = await fetch('/load_documents', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: folderPath })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        ragLoaded = true;
        ragStatusDot.className = 'rag-status-dot loaded';
        ragStatusText.textContent = `${data.files_processed} files, ${data.chunks_created} chunks`;
        
        // Auto-enable RAG mode
        ragEnabledCheckbox.checked = true;
        ragEnabled = true;

        // Show success in chat
        addSystemMessage(`Loaded ${data.files_processed} PDF files (${data.chunks_created} chunks). RAG mode enabled.`, 'success');

    } catch (error) {
        ragStatusDot.className = 'rag-status-dot';
        ragStatusText.textContent = 'Load failed';
        addSystemMessage(`Error loading documents: ${error.message}`, 'error');
    }

    loadBtn.disabled = false;
    loadBtn.textContent = 'Load PDFs';
}

function addSystemMessage(content, type = 'info') {
    const welcome = chatContainer.querySelector('.welcome');
    if (welcome) welcome.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `${type}-message`;
    messageDiv.style.margin = '0.5rem 0';
    messageDiv.textContent = content;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        button.classList.add('copied');
        button.innerHTML = checkIcon + ' Copied';
        setTimeout(() => {
            button.classList.remove('copied');
            button.innerHTML = copyIcon + ' Copy';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

function renderMarkdown(text) {
    // Parse markdown with marked
    let html = marked.parse(text);
    
    // Create a temporary container to manipulate the HTML
    const temp = document.createElement('div');
    temp.innerHTML = html;
    
    // Find all code blocks and wrap them with copy button
    const codeBlocks = temp.querySelectorAll('pre');
    codeBlocks.forEach((pre, index) => {
        const code = pre.querySelector('code');
        const codeText = code ? code.textContent : pre.textContent;
        
        // Detect language from class
        let language = 'code';
        if (code && code.className) {
            const match = code.className.match(/language-(\w+)/);
            if (match) {
                language = match[1];
            }
        }
        
        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        
        // Create header with language and copy button
        const header = document.createElement('div');
        header.className = 'code-block-header';
        header.innerHTML = `
            <span>${language}</span>
            <button class="copy-btn" onclick="copyCodeBlock(this, ${index})">${copyIcon} Copy</button>
        `;
        
        // Store code text as data attribute
        wrapper.setAttribute('data-code', codeText);
        
        // Replace pre with wrapper
        wrapper.appendChild(header);
        wrapper.appendChild(pre.cloneNode(true));
        pre.parentNode.replaceChild(wrapper, pre);
    });
    
    return temp.innerHTML;
}

function copyCodeBlock(button, index) {
    const wrapper = button.closest('.code-block-wrapper');
    const codeText = wrapper.getAttribute('data-code');
    copyToClipboard(codeText, button);
}

function addMessage(role, content, usedRag = false) {
    const welcome = chatContainer.querySelector('.welcome');
    if (welcome) welcome.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = role === 'user' ? 'U' : 'AI';
    let roleLabel = role === 'user' ? 'You' : 'Assistant';
    if (role === 'assistant' && usedRag) {
        roleLabel += '<span class="context-badge">RAG</span>';
    }
    
    // For user messages, escape HTML and preserve whitespace
    // For assistant messages, we'll render markdown later
    const textContent = role === 'user' ? escapeHtml(content) : '';
    const textClass = role === 'assistant' ? 'message-text markdown-content' : 'message-text';
    
    const copyBtn = `<button class="copy-response-btn" onclick="copyFullResponse(this)">${copyIcon} Copy</button>`;
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-header">
                <div class="message-role">${roleLabel}</div>
                ${copyBtn}
            </div>
            <div class="${textClass}">${textContent}</div>
        </div>
    `;
    
    // Store raw content for copying
    messageDiv.setAttribute('data-raw-content', content);
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return messageDiv;
}

function copyFullResponse(button) {
    const message = button.closest('.message');
    const rawContent = message.getAttribute('data-raw-content');
    copyToClipboard(rawContent, button);
}

function addTypingIndicator(usedRag = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'typing-message';
    
    let roleLabel = 'Assistant';
    if (usedRag) {
        roleLabel += '<span class="context-badge">RAG</span>';
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="message-header">
                <div class="message-role">${roleLabel}</div>
                <button class="copy-response-btn" onclick="copyFullResponse(this)">${copyIcon} Copy</button>
            </div>
            <div class="message-text markdown-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return messageDiv;
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isGenerating) return;

    isGenerating = true;
    sendBtn.disabled = true;
    messageInput.value = '';
    messageInput.style.height = 'auto';

    const useRag = ragEnabled && ragLoaded;

    addMessage('user', message);
    
    // Track message for persistence
    conversationMessages.push({ role: 'user', content: message });
    
    const typingDiv = addTypingIndicator(useRag);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message: message,
                session_id: sessionId,
                use_rag: useRag
            })
        });

        if (!response.ok) {
            throw new Error('Failed to get response');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';

        const textDiv = typingDiv.querySelector('.message-text');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.content) {
                            fullResponse += data.content;
                            // Render markdown as we stream
                            textDiv.innerHTML = renderMarkdown(fullResponse);
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                        if (data.error) {
                            textDiv.innerHTML = `<div class="error-message">${escapeHtml(data.error)}</div>`;
                        }
                    } catch (e) {
                        // Ignore parse errors for incomplete chunks
                    }
                }
            }
        }

        // Store raw content for copying
        typingDiv.setAttribute('data-raw-content', fullResponse);
        
        // Track assistant response for persistence
        if (fullResponse) {
            conversationMessages.push({ role: 'assistant', content: fullResponse });
            // Auto-save conversation
            saveConversation();
        }

    } catch (error) {
        const textDiv = typingDiv.querySelector('.message-text');
        textDiv.innerHTML = `<div class="error-message">Error: ${escapeHtml(error.message)}</div>`;
    }

    isGenerating = false;
    sendBtn.disabled = false;
    messageInput.focus();
}

function clearChat() {
    // Start a new conversation instead of just clearing
    newConversation();
}

// Check RAG status on load
async function checkRagStatus() {
    try {
        const response = await fetch('/rag_status');
        const data = await response.json();
        if (data.loaded) {
            ragLoaded = true;
            ragStatusDot.className = 'rag-status-dot loaded';
            ragStatusText.textContent = `${data.chunk_count} chunks loaded`;
        }
    } catch (e) {
        // Ignore
    }
}

// Load available models into dropdown
async function loadModels() {
    try {
        const response = await fetch('/models');
        const data = await response.json();
        
        if (data.error) {
            console.error('Failed to load models:', data.error);
            return;
        }
        
        if (data.models && data.models.length > 0) {
            modelSelect.innerHTML = '';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.name;
                if (model.name === currentModel) {
                    option.selected = true;
                }
                modelSelect.appendChild(option);
            });
        }
    } catch (e) {
        console.error('Error loading models:', e);
    }
}

// Switch to a different model
async function switchModel() {
    const newModel = modelSelect.value;
    if (newModel === currentModel) return;
    
    try {
        const response = await fetch('/switch_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: newModel })
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Failed to switch model: ' + data.error);
            modelSelect.value = currentModel;
            return;
        }
        
        currentModel = newModel;
        addSystemMessage(`Switched to model: ${newModel}`, 'info');
        
    } catch (e) {
        alert('Error switching model: ' + e.message);
        modelSelect.value = currentModel;
    }
}

// Export chat as Markdown
function exportChat() {
    const messages = chatContainer.querySelectorAll('.message');
    if (messages.length === 0) {
        alert('No messages to export.');
        return;
    }
    
    let markdown = `# Chat Export\n\n`;
    markdown += `**Date:** ${new Date().toLocaleString()}\n`;
    markdown += `**Model:** ${currentModel}\n\n---\n\n`;
    
    messages.forEach(msg => {
        const isUser = msg.classList.contains('user');
        const role = isUser ? 'You' : 'Assistant';
        const rawContent = msg.getAttribute('data-raw-content') || '';
        
        markdown += `### ${role}\n\n${rawContent}\n\n`;
    });
    
    // Create and download file
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().slice(0,10)}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    addSystemMessage('Chat exported as Markdown.', 'success');
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Shift + E = Export chat
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'E') {
        e.preventDefault();
        exportChat();
    }
    
    // Ctrl/Cmd + Shift + L = Clear chat
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'L') {
        e.preventDefault();
        clearChat();
    }
    
    // Ctrl/Cmd + Shift + D = Toggle documents panel
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'D') {
        e.preventDefault();
        toggleRagPanel();
    }
    
    // Ctrl/Cmd + Shift + S = Toggle sidebar
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'S') {
        e.preventDefault();
        toggleSidebar();
    }
    
    // Ctrl/Cmd + Shift + N = New conversation
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'N') {
        e.preventDefault();
        newConversation();
    }
    
    // Escape = Focus input / Cancel
    if (e.key === 'Escape') {
        if (document.activeElement !== messageInput) {
            messageInput.focus();
        }
    }
    
    // Ctrl/Cmd + / = Show shortcuts help
    if ((e.ctrlKey || e.metaKey) && e.key === '/') {
        e.preventDefault();
        showShortcutsHelp();
    }
});

function showShortcutsHelp() {
    const shortcuts = `
Keyboard Shortcuts:

Enter - Send message
Shift + Enter - New line
Ctrl + Shift + E - Export chat
Ctrl + Shift + L - Clear chat  
Ctrl + Shift + D - Toggle documents panel
Ctrl + Shift + S - Toggle sidebar
Ctrl + Shift + N - New conversation
Ctrl + / - Show this help
Escape - Focus input
    `.trim();
    
    alert(shortcuts);
}

// ============== Conversation Persistence ==============

// Toggle sidebar visibility
function toggleSidebar() {
    sidebar.classList.toggle('collapsed');
    sidebar.classList.toggle('open');
}

// Load conversation list from server
async function loadConversations() {
    try {
        const response = await fetch('/conversations');
        const data = await response.json();
        
        allConversations = data.conversations || [];
        renderConversationList(allConversations);
    } catch (e) {
        console.error('Failed to load conversations:', e);
    }
}

// Render conversation list in sidebar
function renderConversationList(convos) {
    if (convos.length === 0) {
        conversationList.innerHTML = '<div class="conversation-list-empty">No saved conversations</div>';
        return;
    }
    
    conversationList.innerHTML = convos.map(convo => {
        const isActive = convo.id === currentConversationId;
        const date = convo.updated_at ? new Date(convo.updated_at).toLocaleDateString() : '';
        
        return `
            <div class="conversation-item ${isActive ? 'active' : ''}" 
                 data-id="${convo.id}"
                 onclick="loadConversation('${convo.id}')">
                <div class="conversation-name">${escapeHtml(convo.name)}</div>
                <div class="conversation-meta">
                    <span>${date}</span>
                    <span>${convo.message_count || 0} msgs</span>
                </div>
                <div class="conversation-actions">
                    <button class="conversation-action-btn" onclick="event.stopPropagation(); renameConversation('${convo.id}')" title="Rename">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
                    </button>
                    <button class="conversation-action-btn" onclick="event.stopPropagation(); deleteConversation('${convo.id}')" title="Delete">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

// Filter conversations by search term
function filterConversations() {
    const searchTerm = document.getElementById('sidebar-search').value.toLowerCase();
    const filtered = allConversations.filter(c => 
        c.name.toLowerCase().includes(searchTerm)
    );
    renderConversationList(filtered);
}

// Start a new conversation
function newConversation() {
    currentConversationId = crypto.randomUUID();
    currentConversationName = 'New Chat';
    sessionId = currentConversationId;
    conversationMessages = [];
    
    conversationTitle.textContent = currentConversationName;
    
    chatContainer.innerHTML = `
        <div class="welcome">
            <h2>Start a conversation</h2>
            <p>Messages are processed locally using Ollama.</p>
            <p style="margin-top: 0.5rem; font-size: 0.85rem;">Click "Documents" to load PDFs for document Q&A.</p>
        </div>
    `;
    
    // Clear server-side history
    fetch('/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
    });
    
    // Update sidebar to remove active state
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.remove('active');
    });
    
    messageInput.focus();
}

// Load a conversation
async function loadConversation(conversationId) {
    try {
        const response = await fetch(`/conversations/${conversationId}`);
        const data = await response.json();
        
        if (data.error) {
            addSystemMessage(`Error loading conversation: ${data.error}`, 'error');
            return;
        }
        
        currentConversationId = conversationId;
        currentConversationName = data.name || 'Untitled';
        sessionId = conversationId;
        conversationMessages = data.messages || [];
        
        conversationTitle.textContent = currentConversationName;
        
        // Clear chat and render messages
        chatContainer.innerHTML = '';
        
        if (conversationMessages.length === 0) {
            chatContainer.innerHTML = `
                <div class="welcome">
                    <h2>Start a conversation</h2>
                    <p>Messages are processed locally using Ollama.</p>
                </div>
            `;
        } else {
            conversationMessages.forEach(msg => {
                if (msg.role === 'user') {
                    addMessage('user', msg.content);
                } else if (msg.role === 'assistant') {
                    const msgDiv = addMessage('assistant', '');
                    const textDiv = msgDiv.querySelector('.message-text');
                    textDiv.innerHTML = renderMarkdown(msg.content);
                    msgDiv.setAttribute('data-raw-content', msg.content);
                }
            });
        }
        
        // Update sidebar active state
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.toggle('active', item.dataset.id === conversationId);
        });
        
        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            sidebar.classList.add('collapsed');
            sidebar.classList.remove('open');
        }
        
        messageInput.focus();
        
    } catch (e) {
        addSystemMessage(`Error loading conversation: ${e.message}`, 'error');
    }
}

// Save current conversation
async function saveConversation() {
    if (conversationMessages.length === 0) return;
    
    // Generate name from first user message if still "New Chat"
    if (currentConversationName === 'New Chat' && conversationMessages.length > 0) {
        const firstUserMsg = conversationMessages.find(m => m.role === 'user');
        if (firstUserMsg) {
            currentConversationName = firstUserMsg.content.slice(0, 50) + (firstUserMsg.content.length > 50 ? '...' : '');
            conversationTitle.textContent = currentConversationName;
        }
    }
    
    try {
        await fetch(`/conversations/${currentConversationId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: currentConversationName,
                model: currentModel,
                messages: conversationMessages
            })
        });
        
        // Refresh conversation list
        loadConversations();
        
    } catch (e) {
        console.error('Failed to save conversation:', e);
    }
}

// Rename a conversation
async function renameConversation(conversationId) {
    const convo = allConversations.find(c => c.id === conversationId);
    const currentName = convo ? convo.name : 'Untitled';
    
    const newName = prompt('Enter new name:', currentName);
    if (!newName || newName.trim() === '') return;
    
    try {
        const response = await fetch(`/conversations/${conversationId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName.trim() })
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Error renaming conversation: ' + data.error);
            return;
        }
        
        // Update local state if it's the current conversation
        if (conversationId === currentConversationId) {
            currentConversationName = newName.trim();
            conversationTitle.textContent = currentConversationName;
        }
        
        // Refresh list
        loadConversations();
        
    } catch (e) {
        alert('Error renaming conversation: ' + e.message);
    }
}

// Delete a conversation
async function deleteConversation(conversationId) {
    if (!confirm('Delete this conversation?')) return;
    
    try {
        const response = await fetch(`/conversations/${conversationId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Error deleting conversation: ' + data.error);
            return;
        }
        
        // If deleted current conversation, start new one
        if (conversationId === currentConversationId) {
            newConversation();
        }
        
        // Refresh list
        loadConversations();
        
    } catch (e) {
        alert('Error deleting conversation: ' + e.message);
    }
}

// Edit conversation name in header
function editConversationName() {
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'conversation-title-input';
    input.value = currentConversationName;
    
    const finishEdit = async () => {
        const newName = input.value.trim();
        if (newName && newName !== currentConversationName) {
            currentConversationName = newName;
            
            // Save to server if we have messages
            if (conversationMessages.length > 0) {
                await fetch(`/conversations/${currentConversationId}/rename`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: newName })
                });
                loadConversations();
            }
        }
        
        conversationTitle.textContent = currentConversationName;
        conversationTitle.style.display = '';
    };
    
    input.addEventListener('blur', finishEdit);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            input.blur();
        }
        if (e.key === 'Escape') {
            input.value = currentConversationName;
            input.blur();
        }
    });
    
    conversationTitle.style.display = 'none';
    conversationTitle.parentNode.insertBefore(input, conversationTitle.nextSibling);
    input.focus();
    input.select();
}

// Initialize
messageInput.focus();
checkRagStatus();
loadModels();
loadConversations();

// Set initial conversation ID
currentConversationId = sessionId;

