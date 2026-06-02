// Add event listener for Enter / Shift+Enter handling
document.addEventListener('DOMContentLoaded', () => {
    const promptInput = document.getElementById('promptInput');
    
    promptInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); 
            // Prevent spamming enter if already executing
            if (!isExecuting) {
                handleAction();
            }
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

let currentAbortController = null;
let isExecuting = false;

// Handles the click on the main button
function handleAction() {
    if (isExecuting) {
        stopAgent();
    } else {
        sendMessage();
    }
}

// Toggles the UI state between "Send" and "Stop"
function setButtonState(executing) {
    isExecuting = executing;
    const btn = document.getElementById('actionBtn');
    const icon = document.getElementById('actionIcon');
    
    if (executing) {
        btn.classList.add('stop-mode');
        icon.className = 'fas fa-square'; // Stop icon
    } else {
        btn.classList.remove('stop-mode');
        icon.className = 'fas fa-paper-plane'; // Send icon
    }
}

// Stops the current execution
function stopAgent() {
    if (currentAbortController) {
        currentAbortController.abort(); // This cancels the fetch request
        currentAbortController = null;
    }
}

async function sendMessage() {
    const promptInput = document.getElementById('promptInput');
    const text = promptInput.value.trim();
    
    if (!text) return;

    // Create a new AbortController for this request
    currentAbortController = new AbortController();
    setButtonState(true);

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
            body: JSON.stringify({ prompt: text }),
            signal: currentAbortController.signal // Attach the abort signal here
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
                        if (data.sources && data.sources.length > 0) {
                            let sourcesList = data.sources.map(src => `<li><code>${src}</code></li>`).join('');
                            htmlContent += `
                            <details class="agent-sources">
                                <summary><i class="fas fa-file-alt"></i> Sources</summary>
                                <ul class="source-steps">${sourcesList}</ul>
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
        if (error.name === 'AbortError') {
            responseDiv.innerHTML += `<br><span style="color: #f59e0b;"><i class="fas fa-exclamation-triangle"></i> Execution stopped by user.</span>`;
            
            // Mark the current tool as aborted if it was running
            if (currentToolLi && !currentToolLi.innerHTML.includes('(Completed)')) {
                currentToolLi.innerHTML += ` <span style="color: #f59e0b;">(Aborted)</span>`;
            }
        } else {
            responseDiv.textContent = "Error: Could not connect to the agent backend.";
        }
    } finally {
        // Always revert the button state when the process finishes or is aborted
        setButtonState(false);
        currentAbortController = null;
    }
}

async function indexFolder() {
    const folderInput = document.getElementById('folderInput');
    const statusText = document.getElementById('indexStatus');
    const path = folderInput.value.trim();

    if (!path) return;

    // Clear any text that might have been below the input previously
    statusText.textContent = "";

    // Save the original states so we can revert back
    const originalValue = folderInput.value;
    const originalPlaceholder = folderInput.placeholder;

    // Apply the blue loading state
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

        // Remove loading state
        folderInput.classList.remove('input-loading');

        if (res.ok) {
            // Apply green success state
            folderInput.placeholder = `Successfully indexed files of ${path}`;
            folderInput.classList.add('input-success');
            
            setTimeout(() => {
                folderInput.classList.remove('input-success');
                setTimeout(() => {
                    folderInput.placeholder = originalPlaceholder;
                }, 500); 
            }, 3500);

        } else {
            // Apply red error state (e.g. invalid path)
            folderInput.placeholder = `Error: ${data.detail || 'Could not index folder'}`;
            folderInput.classList.add('input-error');
            
            setTimeout(() => {
                folderInput.classList.remove('input-error');
                setTimeout(() => {
                    folderInput.placeholder = originalPlaceholder;
                    folderInput.value = originalValue; // Restore the typo so the user can fix it
                }, 500); 
            }, 4000);
        }
    } catch (e) {
        // Apply red error state for network crashes
        folderInput.classList.remove('input-loading');
        folderInput.placeholder = "Network error. Make sure the server is running.";
        folderInput.classList.add('input-error');
        
        setTimeout(() => {
            folderInput.classList.remove('input-error');
            setTimeout(() => {
                folderInput.placeholder = originalPlaceholder;
                folderInput.value = originalValue;
            }, 500); 
        }, 4000);
    } finally {
        folderInput.disabled = false;
    }
}

// Make it available globally for the inline HTML onclick handler
window.indexFolder = indexFolder;
