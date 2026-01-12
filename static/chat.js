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

let isGenerating = false;
let currentModel = modelSelect ? modelSelect.value : 'llama3.2';
let sessionId = crypto.randomUUID();
let ragLoaded = false;
let ragEnabled = false;

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

    } catch (error) {
        const textDiv = typingDiv.querySelector('.message-text');
        textDiv.innerHTML = `<div class="error-message">Error: ${escapeHtml(error.message)}</div>`;
    }

    isGenerating = false;
    sendBtn.disabled = false;
    messageInput.focus();
}

function clearChat() {
    sessionId = crypto.randomUUID();
    chatContainer.innerHTML = `
        <div class="welcome">
            <h2>Start a conversation</h2>
            <p>Messages are processed locally using Ollama.</p>
            <p style="margin-top: 0.5rem; font-size: 0.85rem;">Click "Documents" to load PDFs for document Q&A.</p>
        </div>
    `;
    
    fetch('/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
    });
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
Ctrl + / - Show this help
Escape - Focus input
    `.trim();
    
    alert(shortcuts);
}

// Initialize
messageInput.focus();
checkRagStatus();
loadModels();

