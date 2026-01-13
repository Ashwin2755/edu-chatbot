// Simple but reliable EduBot JavaScript
console.log('🚀 EduBot Enhanced JavaScript Loading...');

class EduBotApp {
    constructor() {
        this.currentConversationId = null;
        this.uploadedFile = null;
        this.extractedText = '';
        this.isTyping = false;
        this.theme = localStorage.getItem('theme') || 'light';
        
        console.log('🔧 Initializing EduBot...');
        this.init();
    }

    init() {
        // Wait for DOM to be fully loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    setup() {
        console.log('📡 Setting up event listeners...');
        this.setupEventListeners();
        this.setupTheme();
        this.loadConversations();
        this.focusInput();
        console.log('✅ EduBot Ready!');
    }

    setupEventListeners() {
        // Message input handling
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        
        if (messageInput && sendBtn) {
            console.log('✅ Found message input and send button');
            
            messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            messageInput.addEventListener('input', () => {
                this.autoResizeTextarea(messageInput);
                this.updateSendButton();
            });

            sendBtn.addEventListener('click', () => {
                console.log('📤 Send button clicked');
                this.sendMessage();
            });
        } else {
            console.error('❌ Could not find message input or send button');
        }

        // File upload handling
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                console.log('📁 File selected');
                this.handleFileSelect(e);
            });
        }

        // Theme toggle
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === '/') {
                e.preventDefault();
                this.toggleTheme();
            }
        });
    }

    setupTheme() {
        if (this.theme === 'dark') {
            document.body.setAttribute('data-theme', 'dark');
            const themeIcon = document.getElementById('theme-icon');
            if (themeIcon) {
                themeIcon.className = 'fas fa-sun';
            }
        }
    }

    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    updateSendButton() {
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        
        if (messageInput && sendBtn) {
            const hasContent = messageInput.value.trim().length > 0;
            sendBtn.disabled = !hasContent || this.isTyping;
            sendBtn.style.opacity = hasContent && !this.isTyping ? '1' : '0.5';
        }
    }

    focusInput() {
        setTimeout(() => {
            const messageInput = document.getElementById('messageInput');
            if (messageInput) {
                messageInput.focus();
            }
        }, 100);
    }

    // Message handling
    async sendMessage() {
        console.log('📨 Sending message...');
        const messageInput = document.getElementById('messageInput');
        const message = messageInput ? messageInput.value.trim() : '';
        
        if (!message || this.isTyping) {
            console.log('⚠️ No message or already typing');
            return;
        }

        console.log(`💬 Message: "${message}"`);

        // Set typing state immediately
        this.isTyping = true;
        this.updateSendButton();

        // Clear input and hide welcome screen
        if (messageInput) {
            messageInput.value = '';
            this.autoResizeTextarea(messageInput);
        }
        this.hideWelcomeScreen();

        // Add user message to UI immediately - this should stay visible
        const userMessageElement = this.addMessage('user', message);
        console.log('✅ User message added to UI');

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Prepare request data with proper validation
            const requestData = {
                message: message,
                conversation_id: this.currentConversationId || null,
                context: this.extractedText || "",
                use_history: true,
                temperature: 0.7
            };

            console.log('🌐 Sending request to backend...', requestData);

            // Send request to backend with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('❌ Server error response:', errorText);
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('✅ Received response from backend:', data);
            
            // Update conversation ID
            this.currentConversationId = data.conversation_id;
            
            // Hide typing indicator and add response
            this.hideTypingIndicator();
            
            // Ensure the response is added properly
            if (data.response) {
                this.addMessage('assistant', data.response, data.message_id);
                console.log('✅ Assistant response added to UI');
            } else {
                console.error('❌ No response data received');
                this.addMessage('assistant', 'I apologize, but I received an empty response. Please try asking your question again.');
            }
            
            // Update conversation in sidebar
            this.updateConversationList();
            
        } catch (error) {
            console.error('❌ Error sending message:', error);
            this.hideTypingIndicator();
            
            let errorMessage = 'I apologize, but I encountered an error. ';
            if (error.name === 'AbortError') {
                errorMessage += 'The request timed out. Please try again.';
            } else if (error.message.includes('Failed to fetch')) {
                errorMessage += 'Could not connect to the server. Please check if the server is running.';
            } else {
                errorMessage += `Error: ${error.message}`;
            }
            
            this.addMessage('assistant', errorMessage);
        } finally {
            // Always reset typing state
            this.isTyping = false;
            this.updateSendButton();
        }

        this.focusInput();
    }

    addMessage(role, content, messageId = null) {
        console.log(`➕ Adding ${role} message: "${content.substring(0, 50)}..."`);
        const messagesContainer = document.getElementById('messagesContainer');
        if (!messagesContainer) {
            console.error('❌ Messages container not found');
            return null;
        }

        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role}`;
        
        if (messageId) {
            messageDiv.setAttribute('data-message-id', messageId);
        }

        const avatarIcon = role === 'user' ? '👤' : '🤖';
        
        // Process content for basic formatting
        const processedContent = this.processMarkdown(content);
        
        // Create message HTML with proper structure
        messageDiv.innerHTML = `
            <div class="message-avatar">
                ${avatarIcon}
            </div>
            <div class="message-content">
                <div class="message-text">${processedContent}</div>
                <div class="message-actions">
                    <button class="action-btn" onclick="eduBot.copyMessage(this)" title="Copy">
                        <i class="fas fa-copy"></i>
                    </button>
                    <button class="action-btn" onclick="eduBot.deleteMessage(this)" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;

        // Add to container
        messagesContainer.appendChild(messageDiv);
        
        // Force a repaint to ensure the message is visible
        messageDiv.offsetHeight;
        
        // Add animation class after element is in DOM
        setTimeout(() => {
            messageDiv.classList.add('fade-in');
        }, 10);
        
        // Scroll to bottom
        this.scrollToBottom();
        
        console.log(`✅ ${role} message added successfully`);
        return messageDiv;
    }

    processMarkdown(text) {
        // Basic markdown processing
        let processed = text;
        
        // Bold
        processed = processed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic
        processed = processed.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Code blocks
        processed = processed.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Inline code
        processed = processed.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Line breaks
        processed = processed.replace(/\n/g, '<br>');
        
        return processed;
    }

    showTypingIndicator() {
        console.log('⌨️ Showing typing indicator');
        this.isTyping = true;
        this.updateSendButton();
        
        const messagesContainer = document.getElementById('messagesContainer');
        if (!messagesContainer) return;

        const typingDiv = document.createElement('div');
        typingDiv.className = 'message message-assistant fade-in';
        typingDiv.id = 'typing-indicator';
        
        typingDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="typing-indicator">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        
        messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        console.log('🛑 Hiding typing indicator');
        this.isTyping = false;
        this.updateSendButton();
        
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    hideWelcomeScreen() {
        const welcomeScreen = document.getElementById('welcomeScreen');
        if (welcomeScreen) {
            welcomeScreen.style.display = 'none';
        }
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    // File handling
    triggerFileUpload() {
        console.log('📎 Triggering file upload');
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.click();
        }
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        console.log(`📁 File selected: ${file.name}`);

        // Validate file size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showNotification('File too large. Maximum size is 10MB.', 'error');
            return;
        }

        // Show file preview
        this.showFilePreview(file);

        try {
            const formData = new FormData();
            formData.append('file', file);

            console.log('⬆️ Uploading file...');
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const data = await response.json();
            
            this.uploadedFile = {
                id: data.file_id,
                name: file.name,
                type: file.type
            };
            
            this.extractedText = data.extracted_text;
            
            console.log('✅ File uploaded successfully');
            this.showNotification(`Successfully uploaded: ${file.name}`, 'success');
            
            // Add system message about the upload
            this.addMessage('assistant', `📄 I've successfully processed your file "${file.name}". The document contains ${data.text_length} characters of text. You can now ask me questions about its content!`);
            
        } catch (error) {
            console.error('❌ Upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
            this.removeFile();
        }
    }

    showFilePreview(file) {
        const filePreview = document.getElementById('filePreview');
        const filePreviewText = document.getElementById('filePreviewText');
        
        if (filePreview && filePreviewText) {
            filePreviewText.textContent = `${file.name} (${this.formatFileSize(file.size)})`;
            filePreview.style.display = 'flex';
        }
    }

    removeFile() {
        this.uploadedFile = null;
        this.extractedText = '';
        
        const filePreview = document.getElementById('filePreview');
        const fileInput = document.getElementById('file-input');
        
        if (filePreview) filePreview.style.display = 'none';
        if (fileInput) fileInput.value = '';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Conversation management
    startNewChat() {
        console.log('🆕 Starting new chat');
        this.currentConversationId = null;
        this.clearMessages();
        this.showWelcomeScreen();
        this.removeFile();
        this.updateChatTitle('EduBot Enhanced');
        this.updateConversationList();
        this.focusInput();
    }

    clearMessages() {
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer) {
            const messages = messagesContainer.querySelectorAll('.message:not(#welcomeScreen)');
            messages.forEach(message => message.remove());
        }
    }

    showWelcomeScreen() {
        const welcomeScreen = document.getElementById('welcomeScreen');
        if (welcomeScreen) {
            welcomeScreen.style.display = 'flex';
        }
    }

    updateChatTitle(title) {
        const chatTitle = document.getElementById('chatTitle');
        if (chatTitle) {
            chatTitle.textContent = title;
        }
    }

    async loadConversations() {
        try {
            const response = await fetch('/api/conversations');
            if (response.ok) {
                const data = await response.json();
                this.renderConversationList(data.conversations);
            }
        } catch (error) {
            console.error('Error loading conversations:', error);
        }
    }

    renderConversationList(conversations) {
        const conversationList = document.getElementById('conversationList');
        if (!conversationList) return;

        conversationList.innerHTML = '';

        conversations.forEach(conv => {
            const convDiv = document.createElement('button');
            convDiv.className = 'conversation-item';
            convDiv.textContent = conv.last_message;
            convDiv.onclick = () => this.loadConversation(conv.conversation_id);
            
            if (conv.conversation_id === this.currentConversationId) {
                convDiv.classList.add('active');
            }
            
            conversationList.appendChild(convDiv);
        });
    }

    updateConversationList() {
        this.loadConversations();
    }

    async clearCurrentChat() {
        if (!this.currentConversationId) {
            this.startNewChat();
            return;
        }

        if (confirm('Are you sure you want to clear this conversation?')) {
            try {
                await fetch(`/api/conversations/${this.currentConversationId}`, {
                    method: 'DELETE'
                });
                this.startNewChat();
            } catch (error) {
                console.error('Error clearing conversation:', error);
                this.startNewChat(); // Clear anyway
            }
        }
    }

    // Message actions
    copyMessage(button) {
        const messageContent = button.closest('.message-content').querySelector('.message-text');
        const text = messageContent.textContent || messageContent.innerText;
        
        navigator.clipboard.writeText(text).then(() => {
            this.showNotification('Message copied to clipboard', 'success');
        }).catch(() => {
            this.showNotification('Failed to copy message', 'error');
        });
    }

    deleteMessage(button) {
        if (confirm('Are you sure you want to delete this message?')) {
            const messageElement = button.closest('.message');
            messageElement.style.animation = 'fadeOut 0.3s ease-out';
            setTimeout(() => {
                messageElement.remove();
            }, 300);
        }
    }

    // Theme management
    toggleTheme() {
        console.log('🎨 Toggling theme');
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        localStorage.setItem('theme', this.theme);
        
        if (this.theme === 'dark') {
            document.body.setAttribute('data-theme', 'dark');
            const themeIcon = document.getElementById('theme-icon');
            if (themeIcon) themeIcon.className = 'fas fa-sun';
        } else {
            document.body.removeAttribute('data-theme');
            const themeIcon = document.getElementById('theme-icon');
            if (themeIcon) themeIcon.className = 'fas fa-moon';
        }
    }

    // Utility functions
    showNotification(message, type = 'info') {
        console.log(`📢 Notification: ${message} (${type})`);
        
        // Create notification element
        const notification = document.createElement('div');
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            border-radius: 8px;
            color: white;
            z-index: 1000;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            animation: slideIn 0.3s ease-out;
        `;
        
        if (type === 'success') {
            notification.style.backgroundColor = '#10a37f';
        } else if (type === 'error') {
            notification.style.backgroundColor = '#ef4444';
        } else {
            notification.style.backgroundColor = '#6366f1';
        }
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }

    exportChat() {
        const messages = document.querySelectorAll('.message:not(.typing-indicator)');
        let chatText = '';
        
        messages.forEach(message => {
            const role = message.classList.contains('message-user') ? 'User' : 'Assistant';
            const content = message.querySelector('.message-text').textContent;
            chatText += `${role}: ${content}\n\n`;
        });
        
        if (chatText) {
            const blob = new Blob([chatText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `edubot-chat-${new Date().toISOString().split('T')[0]}.txt`;
            a.click();
            URL.revokeObjectURL(url);
        } else {
            this.showNotification('No messages to export', 'info');
        }
    }

    showSettings() {
        this.showNotification('Settings panel coming soon!', 'info');
    }
}

// Global functions for HTML onclick handlers
let eduBot;

function startNewChat() {
    console.log('🌐 Global: startNewChat called');
    if (eduBot) eduBot.startNewChat();
}

function clearCurrentChat() {
    console.log('🌐 Global: clearCurrentChat called');
    if (eduBot) eduBot.clearCurrentChat();
}

function exportChat() {
    console.log('🌐 Global: exportChat called');
    if (eduBot) eduBot.exportChat();
}

function showSettings() {
    console.log('🌐 Global: showSettings called');
    if (eduBot) eduBot.showSettings();
}

function toggleTheme() {
    console.log('🌐 Global: toggleTheme called');
    if (eduBot) eduBot.toggleTheme();
}

function triggerFileUpload() {
    console.log('🌐 Global: triggerFileUpload called');
    if (eduBot) eduBot.triggerFileUpload();
}

function handleFileSelect(event) {
    console.log('🌐 Global: handleFileSelect called');
    if (eduBot) eduBot.handleFileSelect(event);
}

function removeFile() {
    console.log('🌐 Global: removeFile called');
    if (eduBot) eduBot.removeFile();
}

function sendMessage() {
    console.log('🌐 Global: sendMessage called');
    if (eduBot) eduBot.sendMessage();
}

// Initialize the app
console.log('🎯 Initializing EduBot App...');
eduBot = new EduBotApp();

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    
    .fade-in {
        animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(style);

console.log('🎉 EduBot Enhanced JavaScript Loaded Successfully!');