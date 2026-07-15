// Helper function to escape HTML to prevent XSS
function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;');
}

/**
 * Generates the HTML structure of the user's question/prompt message.
 * @param {string} text The user question text.
 * @returns {string} The HTML string.
 */
function createUserMessageHtml(text) {
    return `
        <div class="message user">
            <div class="bubble">
                ${escapeHtml(text)}
            </div>
        </div>
    `;
}

/**
 * Generates the HTML structure of the AI's response message.
 * @param {string} answerText The main answer text returned by the system.
 * @param {Array<{name: string, page?: string|number}>} [sources=[]] Optional list of source documents used in RAG.
 * @returns {string} The HTML string.
 */
function createAiResponseHtml(answerText, sources = []) {
    let sourcesHtml = '';
    
    if (sources && sources.length > 0) {
        const sourceTags = sources.map(src => {
            const pageSpan = src.page ? ` <span>Pág ${src.page}</span>` : '';
            return `<div class="source-tag">${escapeHtml(src.name)}${pageSpan}</div>`;
        }).join('\n                            ');

        sourcesHtml = `
                    <!-- Bloco de Fontes (RAG) -->
                    <div class="sources-block">
                        <div class="sources-title">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>
                            </svg>
                            Fontes Consultadas
                        </div>
                        <div class="sources-list">
                            ${sourceTags}
                        </div>
                    </div>`;
    }

    const parsedHtml = typeof marked !== 'undefined' ? marked.parse(answerText) : `<p>${escapeHtml(answerText).replace(/\n/g, '<br>')}</p>`;

    return `
        <div class="message ai">
            <div class="avatar">IA</div>
            <div class="bubble">
                ${parsedHtml}
                ${sourcesHtml}
            </div>
        </div>
    `;
}

document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const chatContent = document.getElementById('chat-content');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    // Auto-adjust height of textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
    });

    async function handleSend() {
        const question = chatInput.value.trim();
        if (!question) return;

        // Clear input field and reset height
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // 1. Add user question message
        const userMsgHtml = createUserMessageHtml(question);
        chatContent.insertAdjacentHTML('beforeend', userMsgHtml);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // 2. Add temporary loading/typing indicator bubble
        const loadingId = 'loading-' + Date.now();
        const loadingHtml = `
            <div id="${loadingId}" class="message ai">
                <div class="avatar">IA</div>
                <div class="bubble" style="color: var(--text-muted);">
                    Digitando...
                </div>
            </div>
        `;
        chatContent.insertAdjacentHTML('beforeend', loadingHtml);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            // 3. Fetch RAG response from backend
            const response = await fetch(`/answer?question=${encodeURIComponent(question)}`);
            if (!response.ok) {
                throw new Error('Erro ao obter resposta do servidor.');
            }
            const data = await response.json();

            // Remove loading indicator
            const loadingEl = document.getElementById(loadingId);
            if (loadingEl) loadingEl.remove();

            // Extract sources
            const sources = (data.contexts || []).map(ctx => ({
                name: ctx.metadata?.source || 'Documento',
                page: ctx.metadata?.page || ''
            }));

            // 4. Render AI answer
            const aiMsgHtml = createAiResponseHtml(data.answer, sources);
            chatContent.insertAdjacentHTML('beforeend', aiMsgHtml);
            chatMessages.scrollTop = chatMessages.scrollHeight;

        } catch (error) {
            console.error(error);
            const loadingEl = document.getElementById(loadingId);
            if (loadingEl) loadingEl.remove();

            const errorHtml = `
                <div class="message ai">
                    <div class="avatar">IA</div>
                    <div class="bubble" style="color: #ef4444; border-color: #fca5a5; background-color: #fef2f2;">
                        Erro ao obter resposta do servidor. Por favor, tente novamente mais tarde.
                    </div>
                </div>
            `;
            chatContent.insertAdjacentHTML('beforeend', errorHtml);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    sendBtn.addEventListener('click', handleSend);

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
});
