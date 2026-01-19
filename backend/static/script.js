/**
 * Mega Ai - Client Logic
 */

class EduBot {
    constructor() {
        this.currentConversationId = null;
        this.isTyping = false;
        this.msgContainer = document.getElementById('msgContainer');
        this.userInput = document.getElementById('userInput');
        this.convList = document.getElementById('conversationList');
        this.currentImage = null; // Store base64 image
        this.selectedModel = 'auto'; // 'auto', 'gemini-flash', 'gemini-pro'
        this.sidebar = document.querySelector('.sidebar');

        // Auth State
        this.token = localStorage.getItem('edubot_token');
        this.userName = localStorage.getItem('edubot_user');
        this.authMode = 'login'; // 'login' or 'signup'

        // Configure Marked.js
        marked.setOptions({
            highlight: (code, lang) => {
                if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
                return hljs.highlightAuto(code).value;
            },
            breaks: true
        });

        this.init();
    }

    async init() {
        this.updateAuthUI();
        await this.loadConversations();
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Image input listener
        document.getElementById('fileInput').addEventListener('change', (e) => this.handleFileSelect(e));
    }

    // --- Authentication ---
    updateAuthUI() {
        const loginBtn = document.getElementById('loginBtn');
        const userProfile = document.getElementById('userProfile');
        const userNameDisplay = document.getElementById('userNameDisplay');

        if (this.token) {
            loginBtn.style.display = 'none';
            userProfile.style.display = 'flex';
            userNameDisplay.textContent = this.userName || 'Account';
        } else {
            loginBtn.style.display = 'flex';
            userProfile.style.display = 'none';
        }
    }

    showAuthModal() { window.location.href = '/login'; }
    hideAuthModal() { document.getElementById('authModal').style.display = 'none'; }

    toggleSidebar() {
        this.sidebar.classList.toggle('active');
    }

    startVoiceChat() {
        alert("Voice Interaction is coming in Phase 3! Stay tuned.");
    }

    switchAuthTab(mode) {
        this.authMode = mode;
        const loginTab = document.getElementById('loginTab');
        const signupTab = document.getElementById('signupTab');
        const nameGroup = document.getElementById('nameGroup');
        const modalTitle = document.getElementById('modalTitle');

        if (mode === 'login') {
            loginTab.classList.add('active');
            signupTab.classList.remove('active');
            nameGroup.style.display = 'none';
            modalTitle.textContent = 'Welcome Back';
        } else {
            signupTab.classList.add('active');
            loginTab.classList.remove('active');
            nameGroup.style.display = 'block';
            modalTitle.textContent = 'Create Account';
        }
    }

    async handleAuth(e) {
        e.preventDefault();
        const email = document.getElementById('authEmail').value;
        const password = document.getElementById('authPassword').value;
        const name = document.getElementById('authName').value;

        const url = this.authMode === 'login' ? '/api/auth/login' : '/api/auth/register';
        const body = this.authMode === 'login' ? { email, password } : { email, password, name };

        try {
            const resp = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Auth failed');
            }

            const data = await resp.json();
            this.token = data.access_token;
            this.userName = data.user_name || email.split('@')[0];

            localStorage.setItem('edubot_token', this.token);
            localStorage.setItem('edubot_user', this.userName);

            this.hideAuthModal();
            this.updateAuthUI();
            await this.loadConversations();
        } catch (e) {
            alert(e.message);
        }
    }

    logout() {
        localStorage.removeItem('edubot_token');
        localStorage.removeItem('edubot_user');
        this.token = null;
        this.userName = null;
        this.currentConversationId = null;
        window.location.reload();
    }

    // --- UI Helpers ---
    setModel(model) {
        this.selectedModel = model;
        // Update UI
        document.querySelectorAll('.model-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.model === model);
        });
        const badge = document.getElementById('activeModelBadge');
        if (badge) {
            badge.textContent = model === 'auto' ? 'Auto' : (model === 'gemini-pro' ? 'Pro' : 'Flash');
        }
    }

    exportChat() {
        const title = document.getElementById('sessionTitle').textContent;
        const messages = Array.from(document.querySelectorAll('.message:not(#typing)')).map(msg => {
            const role = msg.dataset.role === 'user' ? 'YOU' : 'MEGA AI';
            const content = msg.querySelector('.content').innerText;
            return `${role}:\n${content}\n\n---\n\n`;
        }).join('');

        if (messages.length === 0) return alert("Nothing to export yet!");

        const blob = new Blob([`MEGA AI STUDY SESSION: ${title}\n\n${messages}`], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `MegaAi-Session-${new Date().toISOString().slice(0, 10)}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    // --- API Helper ---
    async apiFetch(url, options = {}) {
        const headers = options.headers || {};
        if (this.token) headers['Authorization'] = `Bearer ${this.token}`;

        const resp = await fetch(url, { ...options, headers });
        if (resp.status === 401) this.logout();
        return resp;
    }

    // --- History & Chat ---
    async loadConversations() {
        if (!this.token) {
            this.convList.innerHTML = '<p style="padding: 1rem; color: var(--text-muted); font-size: 0.8rem;">Log in to sync history</p>';
            return;
        }
        try {
            const resp = await this.apiFetch('/api/conversations');
            const data = await resp.json();
            this.conversations = data.conversations || [];
            this.renderConversationList();
        } catch (e) { console.error("History load error", e); }
    }

    renderConversationList() {
        if (!this.conversations || this.conversations.length === 0) {
            this.convList.innerHTML = '<p style="padding: 1rem; color: var(--text-muted); font-size: 0.8rem; text-align: center;">No recent sessions</p>';
            return;
        }
        this.convList.innerHTML = '';
        this.conversations.forEach(conv => {
            const el = document.createElement('div');
            el.className = `conv-item ${conv.id === this.currentConversationId ? 'active' : ''}`;

            const titleSpan = document.createElement('span');
            titleSpan.className = 'conv-title';
            titleSpan.textContent = conv.title || 'Untitled Session';

            el.onclick = () => this.loadConversation(conv.id);

            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'icon-btn delete-conv-btn';
            deleteBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                if (confirm('Delete this session permanently?')) this.deleteConversation(conv.id);
            };

            el.innerHTML = `<i class="fas fa-message"></i>`;
            el.appendChild(titleSpan);
            el.appendChild(deleteBtn);
            this.convList.appendChild(el);
        });
    }

    async deleteConversation(id) {
        try {
            await this.apiFetch(`/api/conversations/${id}`, { method: 'DELETE' });
            if (this.currentConversationId === id) this.startNewChat();
            else await this.loadConversations();
        } catch (e) { alert("Failed to delete"); }
    }

    async loadConversation(id) {
        this.currentConversationId = id;
        this.msgContainer.innerHTML = '';

        // Close sidebar on mobile after selection
        if (window.innerWidth <= 768) {
            this.sidebar.classList.remove('active');
        }

        document.querySelector('.welcome-view')?.remove();

        try {
            const resp = await this.apiFetch(`/api/conversations/${id}`);
            const data = await resp.json();

            data.messages.forEach(msg => {
                this.appendMessage(msg.role, msg.content, msg.image_data);
            });

            this.renderConversationList(); // Re-render to update active state

            // Priority: data.title > sidebar text > ID fallback
            const displayTitle = data.title || 'Untitled Session';
            document.getElementById('sessionTitle').textContent = displayTitle;

            this.scrollToBottom();

            // Focus input for continuing conversation
            this.userInput.focus();
        } catch (e) { console.error("Load conversation error", e); }
    }

    async sendMessage(overrideText = null) {
        const text = overrideText || this.userInput.value.trim();
        if (!text && !this.currentImage || this.isTyping) return;

        if (!this.token) {
            this.showAuthModal();
            return;
        }

        this.userInput.value = '';
        this.userInput.style.height = 'auto';

        document.querySelector('.welcome-view')?.remove();

        const imageData = this.currentImage;
        this.appendMessage('user', text, imageData);
        this.removeFile();
        this.scrollToBottom();

        this.showTyping();

        try {
            // Use streaming endpoint for real-time responses
            const resp = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({
                    message: text,
                    image_data: imageData,
                    conversation_id: this.currentConversationId,
                    temperature: 0.7,
                    token: this.token,
                    force_model: this.selectedModel === 'auto' ? null : this.selectedModel
                })
            });

            this.hideTyping();

            // Create assistant message placeholder
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message';
            msgDiv.innerHTML = `
                <div class="avatar avatar-bot"><i class="fas fa-robot"></i></div>
                <div class="content">
                    <div class="message-actions"></div>
                    <div class="content-body"></div>
                </div>`;
            this.msgContainer.appendChild(msgDiv);
            const contentWrap = msgDiv.querySelector('.content-body');
            const actionsWrap = msgDiv.querySelector('.message-actions');

            // Handle streaming response
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.slice(6);
                        try {
                            const data = JSON.parse(jsonStr);

                            if (data.type === 'conversation_id') {
                                const isNew = !this.currentConversationId;
                                this.currentConversationId = data.id;
                                if (isNew) {
                                    // Update title immediately from the message
                                    document.getElementById('sessionTitle').textContent = text.slice(0, 30) + (text.length > 30 ? '...' : '');
                                    this.loadConversations(); // Trigger a background refresh to show it in sidebar
                                }
                            } else if (data.type === 'content') {
                                // Check if the content is an error message from the backend
                                const errorPhrases = [
                                    "I'm having trouble connecting",
                                    "All Gemini models failed",
                                    "API not configured",
                                    "services temporarily unavailable"
                                ];
                                const isErrorContent = errorPhrases.some(phrase => 
                                    data.text && data.text.includes(phrase)
                                );
                                
                                if (isErrorContent) {
                                    contentWrap.innerHTML = `<p class="error-text">⚠️ ${data.text}</p>`;
                                } else {
                                    fullResponse += data.text;
                                    // Update display in real-time
                                    contentWrap.innerHTML = marked.parse(fullResponse);
                                    msgDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
                                    this.scrollToBottom();
                                }
                            } else if (data.type === 'error') {
                                contentWrap.innerHTML = `<p class="error-text">⚠️ ${data.message}</p>`;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE:', e);
                        }
                    }
                }
            }

            // Add action buttons once complete
            if (fullResponse) {
                actionsWrap.innerHTML = `
                    <button class="copy-btn action-icon" title="Copy" onclick="eduBot.copyToClipboard(this.closest('.message').querySelector('.content-body').innerText, this)">
                        <i class="far fa-copy"></i>
                    </button>
                    <button class="action-icon" title="Like"><i class="far fa-thumbs-up"></i></button>
                    <button class="action-icon" title="Dislike"><i class="far fa-thumbs-down"></i></button>
                    <button class="action-icon" title="Share"><i class="fas fa-share-alt"></i></button>
                    <button class="action-icon" title="Regenerate" onclick="eduBot.sendMessage()"><i class="fas fa-redo"></i></button>
                    <button class="action-icon" title="More"><i class="fas fa-ellipsis-h"></i></button>
                `;
                msgDiv.querySelector('.content').classList.add('has-actions');
            }

            this.scrollToBottom();
            await this.loadConversations(); // Final refresh after stream completes
        } catch (e) {
            this.hideTyping();
            this.appendMessage('assistant', "⚠️ Connection lost.");
            console.error('Chat error:', e);
        }
    }

    appendMessage(role, text, image = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message';

        const avatarClass = role === 'user' ? 'avatar-user' : 'avatar-bot';
        const icon = role === 'user' ? 'fas fa-user' : 'fas fa-robot';

        msgDiv.innerHTML = `
            <div class="avatar ${avatarClass}">
                <i class="${icon}"></i>
            </div>
            <div class="content">
                ${role === 'assistant' ? `
                    <div class="message-actions">
                        <button class="copy-btn action-icon" title="Copy" onclick="eduBot.copyToClipboard(this.closest('.message').querySelector('.content-body').innerText, this)">
                            <i class="far fa-copy"></i>
                        </button>
                        <button class="action-icon" title="Like"><i class="far fa-thumbs-up"></i></button>
                        <button class="action-icon" title="Dislike"><i class="far fa-thumbs-down"></i></button>
                        <button class="action-icon" title="Share"><i class="fas fa-share-alt"></i></button>
                        <button class="action-icon" title="Regenerate" onclick="eduBot.sendMessage()"><i class="fas fa-redo"></i></button>
                        <button class="action-icon" title="More"><i class="fas fa-ellipsis-h"></i></button>
                    </div>
                ` : ''}
                <div class="content-body">
                    ${text ? (role === 'assistant' ? marked.parse(text) : `<p>${this.escape(text)}</p>`) : ''}
                </div>
                ${image ? `<img src="data:image/jpeg;base64,${image}" class="chat-image">` : ''}
            </div>
        `;

        this.msgContainer.appendChild(msgDiv);

        // Re-highlight if needed
        if (role === 'assistant') {
            msgDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
        }
        this.scrollToBottom();
    }

    showTyping() {
        this.isTyping = true;
        const el = document.createElement('div');
        el.id = 'typing'; el.className = 'message';
        el.innerHTML = `<div class="avatar avatar-bot"><i class="fas fa-robot"></i></div><div class="content"><p>Thinking...</p></div>`;
        this.msgContainer.appendChild(el);
        this.scrollToBottom();
    }

    hideTyping() { this.isTyping = false; document.getElementById('typing')?.remove(); }

    scrollToBottom() { this.msgContainer.scrollTop = this.msgContainer.scrollHeight; }

    async copyToClipboard(text, btn) {
        try {
            await navigator.clipboard.writeText(text);
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-check"></i>';
            btn.classList.add('copied');

            setTimeout(() => {
                btn.innerHTML = originalHTML;
                btn.classList.remove('copied');
            }, 2000);
        } catch (err) {
            console.error('Failed to copy: ', err);
            alert('Failed to copy text.');
        }
    }

    escape(text) {
        if (!text) return "";
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    autoGrow(el) { el.style.height = "5px"; el.style.height = (el.scrollHeight) + "px"; }

    // File & Image Handling
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;

        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                this.currentImage = ev.target.result.split(',')[1];
                document.getElementById('fileName').textContent = file.name;
                document.getElementById('uploadPreview').style.display = 'flex';
            };
            reader.readAsDataURL(file);
        } else {
            this.handleFileUpload(file);
        }
    }

    async handleFileUpload(file) {
        if (!this.token) { this.showAuthModal(); return; }
        const fileNameLabel = document.getElementById('fileName');
        const uploadPreview = document.getElementById('uploadPreview');

        fileNameLabel.textContent = file.name;
        uploadPreview.style.display = 'flex';

        const formData = new FormData();
        formData.append('file', file);

        try {
            await this.apiFetch('/api/upload', { method: 'POST', body: formData });
            this.appendMessage('assistant', `✅ **File Context Added**: I've indexed \`${file.name}\`. You can now ask questions about it.`);
        } catch (e) {
            alert("Upload failed");
            this.removeFile();
        }
    }

    removeFile() {
        this.currentImage = null;
        document.getElementById('uploadPreview').style.display = 'none';
        document.getElementById('fileInput').value = '';
    }

    exportChat() {
        const content = Array.from(this.msgContainer.querySelectorAll('.message'))
            .map(m => `[${m.querySelector('.avatar-user') ? 'User' : 'Assistant'}]\n${m.querySelector('.content').innerText}\n`).join('\n---\n\n');
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([content], { type: 'text/markdown' }));
        a.download = `edu-export-${Date.now()}.md`; a.click();
    }

    startNewChat() {
        this.currentConversationId = null;
        this.msgContainer.innerHTML = `
            <div class="welcome-view">
                <div class="welcome-center">
                    <h2>Your Mega Ai Learning Assistant.</h2>
                    <p>Advanced tutoring for code, documents, and deep analysis.</p>
                </div>
                
                <div class="welcome-cards">
                    <div class="welcome-card" onclick="document.getElementById('fileInput').click()">
                        <i class="fas fa-file-upload"></i>
                        <h3>Upload file</h3>
                    </div>
                    <div class="welcome-card" onclick="eduBot.sendMessage('Create an image showing...')">
                        <i class="fas fa-palette"></i>
                        <h3>Create image</h3>
                    </div>
                    <div class="welcome-card" onclick="eduBot.sendMessage('Search the web for...')">
                        <i class="fas fa-globe"></i>
                        <h3>Web search</h3>
                    </div>
                    <div class="welcome-card" onclick="eduBot.startVoiceChat()">
                        <i class="fas fa-microphone"></i>
                        <h3>Start voice chat</h3>
                    </div>
                </div>
            </div>
        `;
        document.getElementById('sessionTitle').textContent = 'New Mega Ai Session';
        this.loadConversations(); // Proactively refresh list
        this.userInput.value = '';
        this.userInput.style.height = 'auto';
        this.userInput.focus();
    }
}

// Global initialization
window.eduBot = new EduBot();