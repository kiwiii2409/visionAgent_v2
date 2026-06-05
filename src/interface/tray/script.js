let currentAbortController = null;
let isExecuting = false;

document.addEventListener('DOMContentLoaded', () => {
    // DOM bindings
    const promptInput = document.getElementById('promptInput');
    const settingsBtn = document.getElementById('settingsBtn');
    const actionBtn = document.getElementById('actionBtn');
    const indexBtn = document.getElementById('indexBtn');

    // Event Listeners
    settingsBtn.addEventListener('click', toggleSettings);
    actionBtn.addEventListener('click', handleAction);
    indexBtn.addEventListener('click', indexFolder);

    promptInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); 
            if (!isExecuting) handleAction();
        }
    });

    promptInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });
});

function toggleSettings() {
    const panel = document.getElementById('settingsPanel');
    const btn = document.getElementById('settingsBtn');
    
    panel.classList.toggle('hidden');
    btn.classList.toggle('active');
}

function handleAction() {
    if (isExecuting) stopAgent();
    else sendMessage();
}

function setButtonState(executing) {
    isExecuting = executing;
    const btn = document.getElementById('actionBtn');
    const icon = document.getElementById('actionIcon');
    
    if (executing) {
        btn.classList.add('stop-mode');
        icon.className = 'fas fa-square';
    } else {
        btn.classList.remove('stop-mode');
        icon.className = 'fas fa-paper-plane'; 
    }
}

function stopAgent() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
}

function createMessageBubble(sender) {
    const chatHistory = document.getElementById('chatHistory');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    msgDiv.appendChild(contentDiv);
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    return contentDiv;
}

async function sendMessage() {
    const promptInput = document.getElementById('promptInput');
    const text = promptInput.value.trim();
    if (!text) return;

    currentAbortController = new AbortController();
    setButtonState(true);

    const userBubble = createMessageBubble('user');
    userBubble.textContent = text;
    
    promptInput.value = '';
    promptInput.style.height = 'auto';
    promptInput.focus();

    // Auto-hide settings
    document.getElementById('settingsPanel').classList.add('hidden');
    document.getElementById('settingsBtn').classList.remove('active');

    const systemBubble = createMessageBubble('system');
    systemBubble.innerHTML = `
        <details class="agent-thinking" open>
            <summary><i class="fas fa-terminal"></i> Execution Log</summary>
            <ul class="thinking-steps"></ul>
        </details>
        <div class="agent-response"></div>
    `;
    
    const thinkingToggle = systemBubble.querySelector('.agent-thinking');
    const thinkingSteps = systemBubble.querySelector('.thinking-steps');
    const responseDiv = systemBubble.querySelector('.agent-response');
    let currentToolLi = null;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: text }),
            signal: currentAbortController.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            buffer = lines.pop(); 

            for (let line of lines) {
                if (!line.trim()) continue;
                try {
                    const data = JSON.parse(line);
                    
                    if (data.type === "init") {
                        const li = document.createElement('li');
                        li.innerHTML = `<span style="color: var(--primary);">System routing to <b>${data.mode.toUpperCase()}</b> execution mode.</span>`;
                        thinkingSteps.appendChild(li);
                    } 
                    else if (data.type === "tool") {
                        currentToolLi = document.createElement('li');
                        currentToolLi.innerHTML = `Executing: <code>${data.name}</code>`;
                        thinkingSteps.appendChild(currentToolLi);
                    } 
                    else if (data.type === "tool_done" && currentToolLi) {
                        currentToolLi.innerHTML += ` (Completed)`;
                    } 
                    else if (data.type === "msg") {
                        let htmlContent = marked.parse(data.content);
                        if (data.sources && data.sources.length > 0) {
                            const sourcesList = data.sources.map(src => `<li><code>${src}</code></li>`).join('');
                            htmlContent += `
                            <details class="agent-sources">
                                <summary><i class="fas fa-file-alt"></i> Sources</summary>
                                <ul class="source-steps list-unstyled">${sourcesList}</ul>
                            </details>`;
                        }
                        responseDiv.innerHTML = htmlContent;
                        thinkingToggle.removeAttribute('open');
                    } 
                    else if (data.type === "error") {
                        responseDiv.innerHTML += `<br><span class="text-error">Error: ${data.content}</span>`;
                    }
                    document.getElementById('chatHistory').scrollTop = document.getElementById('chatHistory').scrollHeight;
                } catch (err) {
                    console.error("Failed to parse stream chunk:", err, line);
                }
            }
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            responseDiv.innerHTML += `<br><span class="text-warning"><i class="fas fa-exclamation-triangle"></i> Execution stopped.</span>`;
            if (currentToolLi && !currentToolLi.innerHTML.includes('(Completed)')) {
                currentToolLi.innerHTML += ` <span class="text-warning">(Aborted)</span>`;
            }
        } else {
            responseDiv.textContent = "Error: Could not connect to backend.";
        }
    } finally {
        setButtonState(false);
        currentAbortController = null;
    }
}

async function indexFolder() {
    const folderInput = document.getElementById('folderInput');
    const path = folderInput.value.trim();
    if (!path) return;

    const originalValue = folderInput.value;
    const originalPlaceholder = folderInput.placeholder;

    folderInput.value = "";
    folderInput.placeholder = "Indexing...";
    folderInput.classList.add('input-loading');
    folderInput.disabled = true;

    try {
        const res = await fetch('/api/index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: path })
        });
        const data = await res.json();

        folderInput.classList.remove('input-loading');

        if (res.ok) {
            folderInput.placeholder = `Indexed: ${path}`;
            folderInput.classList.add('input-success');
            setTimeout(() => {
                folderInput.classList.remove('input-success');
                folderInput.placeholder = originalPlaceholder;
            }, 3000);
        } else {
            folderInput.placeholder = `Error: ${data.detail || 'Failed'}`;
            folderInput.classList.add('input-error');
            setTimeout(() => {
                folderInput.classList.remove('input-error');
                folderInput.placeholder = originalPlaceholder;
                folderInput.value = originalValue; 
            }, 4000);
        }
    } catch (e) {
        folderInput.classList.remove('input-loading');
        folderInput.placeholder = "Network error.";
        folderInput.classList.add('input-error');
        setTimeout(() => {
            folderInput.classList.remove('input-error');
            folderInput.placeholder = originalPlaceholder;
            folderInput.value = originalValue;
        }, 4000);
    } finally {
        folderInput.disabled = false;
    }
}