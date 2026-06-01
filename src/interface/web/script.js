// Add event listener for Enter / Shift+Enter handling
document.addEventListener('DOMContentLoaded', () => {
    const promptInput = document.getElementById('promptInput');
    
    promptInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); 
            sendMessage();
        }
    });

    // Auto-grow the textarea as the user types
    promptInput.addEventListener('input', function() {
        // Reset height to auto to recalculate shrinking if user deletes text
        this.style.height = 'auto';
        // Set height to the actual scroll height
        this.style.height = this.scrollHeight + 'px';
    });
});

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

    const userBubble = createMessageBubble('user');
    userBubble.textContent = text;
    
    promptInput.value = '';
    promptInput.style.height = 'auto';

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
            body: JSON.stringify({ prompt: text }) 
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
                        currentToolLi.innerHTML = `Executing Tool: <code>${data.name}</code>`;
                        thinkingSteps.appendChild(currentToolLi);
                    } 
                    else if (data.type === "tool_done" && currentToolLi) {
                        currentToolLi.innerHTML += ` (Completed)`;
                    } 
                    else if (data.type === "msg") {
                        let htmlContent = marked.parse(data.content);
                        
                        // Render structured sources collapsible if they exist
                        if (data.sources && data.sources.length > 0) {
                            let sourcesList = data.sources.map(src => `<li><code>${src}</code></li>`).join('');
                            htmlContent += `
                            <details class="agent-sources">
                                <summary><i class="fas fa-file-alt"></i> Sources</summary>
                                <ul class="source-steps">
                                    ${sourcesList}
                                </ul>
                            </details>`;
                        }
                        
                        responseDiv.innerHTML = htmlContent;
                        thinkingToggle.removeAttribute('open');
                    } 
                    else if (data.type === "error") {
                        responseDiv.innerHTML += `<br><span style="color: #ef4444;">Error: ${data.content}</span>`;
                    }
                    
                    document.getElementById('chatHistory').scrollTop = document.getElementById('chatHistory').scrollHeight;

                } catch (err) {
                    console.error("Failed to parse stream chunk:", err, line);
                }
            }
        }
    } catch (error) {
        responseDiv.textContent = "Error: Could not connect to the agent backend.";
    }
}

async function indexFolder() {
    const folderInput = document.getElementById('folderInput');
    const indexBtn = document.getElementById('indexBtn');
    const statusEl = document.getElementById('indexStatus');
    const folderPath = folderInput?.value?.trim();

    if (!folderPath) {
        statusEl.textContent = 'Please enter a folder path.';
        statusEl.style.color = '#ef4444';
        return;
    }

    indexBtn.disabled = true;
    indexBtn.textContent = 'Indexing...';
    statusEl.textContent = '';

    try {
        const resp = await fetch('/api/index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: folderPath })
        });
        const data = await resp.json();
        if (resp.ok) {
            statusEl.textContent = data.message || 'Indexing complete.';
            statusEl.style.color = '#22c55e';
        } else {
            statusEl.textContent = data.detail || 'Indexing failed.';
            statusEl.style.color = '#ef4444';
        }
    } catch (e) {
        statusEl.textContent = 'Network error while indexing.';
        statusEl.style.color = '#ef4444';
    } finally {
        indexBtn.disabled = false;
        indexBtn.textContent = 'Index';
    }
}