/**
 * Chatbot Widget with Multi-Model Support
 * Supports: GPT-4, GPT-3.5, Claude, Gemini
 */

class ChatbotWidget {
    constructor() {
        this.isOpen = false;
        this.currentModel = 'deepseek';  // Default to DeepSeek
        this.messages = [];
        this.isTyping = false;
        
        this.init();
    }
    
    init() {
        this.createWidget();
        this.bindEvents();
        this.addWelcomeMessage();
    }
    
    createWidget() {
        const container = document.createElement('div');
        container.className = 'chatbot-container';
        container.innerHTML = `
            <div class="chatbot-window">
                <div class="chatbot-header">
                    <div class="chatbot-header-left">
                        <div class="chatbot-avatar">
                            <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
                        </div>
                        <div class="chatbot-header-info">
                            <h3>AI Assistant</h3>
                            <span id="chatbot-status">Online</span>
                        </div>
                    </div>
                    <div class="model-selector">
                        <select id="model-select">
                            <option value="deepseek" selected>DeepSeek</option>
                            <option value="qwen3-max">Qwen3-Max</option>
                            <option value="gpt-4">GPT-4</option>
                        </select>
                    </div>
                </div>
                
                <div class="chatbot-messages" id="chatbot-messages">
                    <!-- Messages will be inserted here -->
                </div>
                
                <div class="chatbot-input">
                    <textarea 
                        id="chatbot-textarea" 
                        placeholder="Ask about your portfolio..." 
                        rows="1"
                    ></textarea>
                    <button class="chatbot-send" id="chatbot-send">
                        <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                    </button>
                </div>
            </div>
            
            <button class="chatbot-toggle" id="chatbot-toggle">
                <svg viewBox="0 0 24 24" id="chat-icon"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/></svg>
                <svg viewBox="0 0 24 24" id="close-icon" style="display:none"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
            </button>
        `;
        
        document.body.appendChild(container);
        
        // Cache DOM elements
        this.container = container;
        this.window = container.querySelector('.chatbot-window');
        this.toggleBtn = container.querySelector('#chatbot-toggle');
        this.messagesContainer = container.querySelector('#chatbot-messages');
        this.textarea = container.querySelector('#chatbot-textarea');
        this.sendBtn = container.querySelector('#chatbot-send');
        this.modelSelect = container.querySelector('#model-select');
        this.chatIcon = container.querySelector('#chat-icon');
        this.closeIcon = container.querySelector('#close-icon');
        this.statusEl = container.querySelector('#chatbot-status');
    }
    
    bindEvents() {
        // Toggle chat window
        this.toggleBtn.addEventListener('click', () => this.toggle());
        
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Enter to send (Shift+Enter for new line)
        this.textarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.textarea.addEventListener('input', () => {
            this.textarea.style.height = 'auto';
            this.textarea.style.height = Math.min(this.textarea.scrollHeight, 120) + 'px';
        });
        
        // Model change
        this.modelSelect.addEventListener('change', (e) => {
            this.currentModel = e.target.value;
        });
    }
    
    toggle() {
        this.isOpen = !this.isOpen;
        this.window.classList.toggle('open', this.isOpen);
        this.toggleBtn.classList.toggle('active', this.isOpen);
        
        // Toggle icons
        this.chatIcon.style.display = this.isOpen ? 'none' : 'block';
        this.closeIcon.style.display = this.isOpen ? 'block' : 'none';
        
        if (this.isOpen) {
            this.textarea.focus();
        }
    }
    
    addWelcomeMessage() {
        this.addBotMessage(
            "👋 Hi! I'm your AI assistant. I can help you analyze your portfolio, " +
            "explain market trends, or answer trading questions. What would you like to know?",
            false
        );
    }
    
    addSystemMessage(text) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message system';
        messageEl.innerHTML = `
            <div style="
                text-align: center;
                font-size: 12px;
                color: #94a3b8;
                padding: 8px;
                width: 100%;
            ">
                ${text}
            </div>
        `;
        this.messagesContainer.appendChild(messageEl);
        this.scrollToBottom();
    }
    
    addUserMessage(text) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message user';
        messageEl.innerHTML = `
            <div class="message-content">${this.escapeHtml(text)}</div>
            <div class="message-avatar">
                <svg viewBox="0 0 24 24"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>
            </div>
        `;
        this.messagesContainer.appendChild(messageEl);
        this.messages.push({ role: 'user', content: text });
        this.scrollToBottom();
    }
    
    addBotMessage(text, showModel = true) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message bot';
        messageEl.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
            </div>
            <div>
                <div class="message-content">${this.formatMessage(text)}</div>
                ${showModel ? `<span class="model-badge">${this.getModelDisplayName(this.currentModel)}</span>` : ''}
            </div>
        `;
        this.messagesContainer.appendChild(messageEl);
        this.messages.push({ role: 'assistant', content: text });
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        const typingEl = document.createElement('div');
        typingEl.className = 'message bot';
        typingEl.id = 'typing-indicator';
        typingEl.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
            </div>
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        this.messagesContainer.appendChild(typingEl);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        const typingEl = document.getElementById('typing-indicator');
        if (typingEl) typingEl.remove();
    }
    
    async sendMessage() {
        const text = this.textarea.value.trim();
        if (!text || this.isTyping) return;
        
        // Clear input
        this.textarea.value = '';
        this.textarea.style.height = 'auto';
        
        // Add user message
        this.addUserMessage(text);
        
        // Show typing indicator
        this.isTyping = true;
        this.sendBtn.disabled = true;
        this.statusEl.textContent = 'Typing...';
        this.showTypingIndicator();
        
        try {
            // Call backend API
            const response = await this.callChatAPI(text);
            this.hideTypingIndicator();
            this.addBotMessage(response);
        } catch (error) {
            this.hideTypingIndicator();
            this.addBotMessage("Sorry, I encountered an error. Please try again.");
            console.error('Chat error:', error);
        } finally {
            this.isTyping = false;
            this.sendBtn.disabled = false;
            this.statusEl.textContent = 'Online';
        }
    }
    
    async callChatAPI(message) {
        // Call backend API endpoint
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                model: this.currentModel,
                history: this.messages.slice(-10) // Last 10 messages for context
            })
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const data = await response.json();
        return data.response;
    }
    
    getModelDisplayName(model) {
        const names = {
            'gpt-4': 'GPT-4',
            'gpt-3.5-turbo': 'GPT-3.5',
            'claude-3-opus': 'Claude 3 Opus',
            'claude-3-sonnet': 'Claude 3 Sonnet',
            'gemini-pro': 'Gemini Pro',
            'llama-3': 'Llama 3'
        };
        return names[model] || model;
    }
    
    formatMessage(text) {
        // Convert markdown-like formatting to HTML
        return this.escapeHtml(text)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
}

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new ChatbotWidget();
});

