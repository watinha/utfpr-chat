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
    // Initialize background canvas animation
    function initAurora() {
        const canvas = document.getElementById('bg-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;

        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });

        const count = 4;
        const colors = [
            'rgba(234, 179, 8, ',   // UTFPR Yellow-500 prefix
            'rgba(245, 158, 11, ',  // UTFPR Amber-500 prefix
            'rgba(253, 224, 71, ',  // UTFPR Yellow-300 prefix
            'rgba(202, 138, 4, '    // UTFPR Dark Gold prefix
        ];

        const waveParams = [];
        for (let i = 0; i < count; i++) {
            waveParams.push({
                pulseSpeed: 0.0008 + Math.random() * 0.001,
                pulseOffset: Math.random() * Math.PI * 2,
                baseOpacity: 0.05 + Math.random() * 0.04
            });
        }

        let phase = 0;

        function draw() {
            const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
            bgGrad.addColorStop(0, '#0b0f19');
            bgGrad.addColorStop(1, '#020617');
            ctx.fillStyle = bgGrad;
            ctx.fillRect(0, 0, width, height);

            ctx.globalCompositeOperation = 'screen';
            phase += 0.002;
            const time = Date.now();

            for (let i = 0; i < count; i++) {
                ctx.beginPath();
                
                const p = waveParams[i];
                const opacity = p.baseOpacity + (Math.sin(time * p.pulseSpeed + p.pulseOffset) * 0.03);
                if (opacity <= 0) continue;

                const colorBase = colors[i];
                const colorStr = colorBase + opacity + ')';
                const transparentStr = colorBase + '0)';

                const gradient = ctx.createLinearGradient(0, 0, width, 0);
                gradient.addColorStop(0, transparentStr);
                gradient.addColorStop(0.5, colorStr);
                gradient.addColorStop(1, transparentStr);
                
                ctx.strokeStyle = gradient;
                ctx.lineWidth = height * 0.25;
                ctx.lineCap = 'round';
                ctx.shadowBlur = 80 * (opacity / 0.08);
                ctx.shadowColor = colorStr;

                for (let x = 0; x <= width; x += 10) {
                    const angle1 = (x * 0.002) + phase + (i * 20);
                    const angle2 = (x * 0.005) - phase * 0.5 + (i * 10);
                    const y = (height * 0.28) + 
                              Math.sin(angle1) * (height * 0.08) + 
                              Math.cos(angle2) * (height * 0.04);
                    
                    if (x === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }

            ctx.globalCompositeOperation = 'source-over';
            ctx.shadowBlur = 0;

            requestAnimationFrame(draw);
        }

        draw();
    }

    initAurora();

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

        // Re-align to only show one Q&A at a time by fading out and removing previous messages
        const existingMessages = Array.from(chatContent.querySelectorAll('.message'));
        if (existingMessages.length > 0) {
            existingMessages.forEach(msg => msg.classList.add('fade-out'));
            // Wait for 1s transition to finish
            await new Promise(resolve => setTimeout(resolve, 1000));
            existingMessages.forEach(msg => msg.remove());
        }

        // 1. Add user question message
        const userMsgHtml = createUserMessageHtml(question);
        chatContent.insertAdjacentHTML('beforeend', userMsgHtml);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // 2. Add temporary loading/typing indicator bubble (Gemini-style skeleton shimmer)
        const loadingId = 'loading-' + Date.now();
        const loadingHtml = `
            <div id="${loadingId}" class="message ai">
                <div class="avatar">IA</div>
                <div class="bubble" style="width: 100%;">
                    <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span>Pesquisando...</span>
                    </div>
                    <div class="skeleton-loader">
                        <div class="skeleton-line"></div>
                        <div class="skeleton-line"></div>
                        <div class="skeleton-line"></div>
                    </div>
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
