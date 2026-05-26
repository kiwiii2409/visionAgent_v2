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
    
    if (!text) return;

    // 1. Show User Message (Removed the manual [MODE] prefix)
    const userBubble = createMessageBubble('user');
    userBubble.textContent = text;
    
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
            // THE FIX: Only send the prompt! The backend handles the routing now.
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
                        // This will now display the mode chosen by your LLM Router!
                        li.innerHTML = `<span style="color: var(--primary);">Router selected <b>${data.mode.toUpperCase()}</b> mode...</span>`;
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
                        responseDiv.innerHTML = marked.parse(data.content);
                        thinkingToggle.removeAttribute('open');
                    } 
                    else if (data.type === "error") {
                        responseDiv.innerHTML += `<br><span style="color: #ef4444;">❌ Error: ${data.content}</span>`;
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