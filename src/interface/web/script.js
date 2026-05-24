// Creates a message bubble and returns the DOM element so we can update it in real-time
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
    const mode = document.getElementById('modeSelector').value;
    if (!text) return;

    // 1. Show User Message
    const userBubble = createMessageBubble('user');
    userBubble.textContent = `[${mode.toUpperCase()}] ${text}`;
    
    promptInput.value = '';
    promptInput.style.height = 'auto';

    // 2. Create the System Bubble with the Collapsible "Thinking" Structure
    const systemBubble = createMessageBubble('system');
    systemBubble.innerHTML = `
        <details class="agent-thinking" open>
            <summary><i class="fas fa-brain"></i> Agent Thinking Process</summary>
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
            body: JSON.stringify({ prompt: text, mode: mode })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = ""; // Buffer to handle split JSON chunks

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Split by newline and process complete JSON strings
            let lines = buffer.split('\n');
            buffer = lines.pop(); // Keep the last incomplete line in the buffer

            for (let line of lines) {
                if (!line.trim()) continue;
                
                try {
                    const data = JSON.parse(line);
                    
                    if (data.type === "init") {
                        const li = document.createElement('li');
                        li.innerHTML = `<span style="color: var(--primary);">Initializing <b>${data.mode.toUpperCase()}</b> mode...</span>`;
                        thinkingSteps.appendChild(li);
                    } 
                    else if (data.type === "tool") {
                        currentToolLi = document.createElement('li');
                        currentToolLi.innerHTML = `🛠️ Executing Tool: <code>${data.name}</code>...`;
                        thinkingSteps.appendChild(currentToolLi);
                    } 
                    else if (data.type === "tool_done" && currentToolLi) {
                        currentToolLi.innerHTML += ` ✅`;
                    } 
                    else if (data.type === "msg") {
                        // Insert final response and CLOSE the thinking toggle!
                        responseDiv.textContent = data.content;
                        thinkingToggle.removeAttribute('open');
                    } 
                    else if (data.type === "error") {
                        responseDiv.innerHTML += `<br><span style="color: #ef4444;">❌ Error: ${data.content}</span>`;
                    }
                    
                    // Keep auto-scrolling to bottom
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