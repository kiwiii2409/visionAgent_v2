import RFB from 'https://cdn.jsdelivr.net/npm/@novnc/novnc@1.4.0/core/rfb.js';

document.addEventListener("DOMContentLoaded", () => {
    // ==============================================
    // 1. DOM Elements Selection
    // ==============================================
    
    // Search Interface Elements
    const searchInterface = document.getElementById("search-interface");
    const searchInput = document.getElementById("search-input");
    const btnSearch = document.getElementById("btn-search");
    const btnTaskMode = document.getElementById("btn-task-mode");
    const resultsContainer = document.getElementById("results-container");
    const aiText = document.getElementById("ai-text");
    const docsList = document.getElementById("docs-list");

    // Task Workspace Elements
    const taskInterface = document.getElementById("task-interface");
    const chatHistory = document.getElementById("chatHistory");
    const promptInput = document.getElementById("promptInput");
    const actionBtn = document.getElementById("actionBtn");
    const actionIcon = document.getElementById("actionIcon");
    const btnBackSearch = document.getElementById("btn-back-search");

    // Settings Elements (Search & Task views)
    const taskModeSelect = document.getElementById("taskModeSelect");
    const searchModeSelect = document.getElementById("searchModeSelect");
    const searchSettingsBtn = document.getElementById("search-settings-btn");
    const searchSettingsDrawer = document.getElementById("search-settings-drawer");
    const searchSettingsOverlay = document.getElementById("search-settings-overlay");
    const closeSearchSettingsBtn = document.getElementById("close-search-settings");

    // ==============================================
    // 2. Interface Toggling Logic
    // ==============================================

    // Switch to Task Mode
    btnTaskMode.addEventListener("click", () => {
        searchInterface.style.display = "none";
        taskInterface.style.display = "flex"; // Uses flex to maintain full screen height
    });

    // Switch back to Search Mode
    btnBackSearch.addEventListener("click", () => {
        taskInterface.style.display = "none";
        searchInterface.style.display = "flex"; 
    });


    // ==============================================
    // 3. Search Engine Logic (NDJSON Streaming)
    // ==============================================

    const performSearch = async () => {
        const query = searchInput.value.trim();
        if (!query) return;

        // Trigger CSS transition (moves search bar to the top)
        searchInterface.classList.add("results-mode");
        resultsContainer.style.display = "block";
        
        // UI Loading State
        aiText.innerHTML = '<span style="color:#a1a1aa;"><em>Connecting to local context...</em></span>';
        docsList.innerHTML = "";

        try {
            const response = await fetch("/api/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query })
            });
            
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop(); // keep the trailing incomplete line
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.type === "tool") {
                            aiText.innerHTML = `<span style="color:#6b7280;"><em>${data.name} <i class="fas fa-spinner fa-spin"></i></em></span>`;
                        } else if (data.type === "msg") {
                            // Render AI Overview
                            if (typeof marked !== 'undefined') {
                                aiText.innerHTML = marked.parse(data.content);
                            } else {
                                aiText.innerText = data.content;
                            }

                            // Render Documents
                            if (data.sources && data.sources.length > 0) {
                                docsList.innerHTML = data.sources.map(doc => `
                                    <div class="doc-result">
                                        <div class="doc-path">${doc.path || doc.source || ''}</div>
                                    <a href="/api/file?path=${encodeURIComponent(doc.path || doc.source)}" target="_blank" rel="noopener noreferrer" class="doc-title">${doc.name || doc.path || 'Document'}</a>
                                        <div class="doc-summary">${doc.summary || ''}</div>
                                    </div>
                                `).join('');
                            } else {
                                docsList.innerHTML = `<p style="color: #6b7280;">No explicit documents cited.</p>`;
                            }
                        } else if (data.type === "error") {
                            aiText.innerHTML = `<span style="color: #ef4444;">${data.content}</span>`;
                        }
                    } catch (e) {
                        console.warn("Could not parse JSON chunk:", line, e);
                    }
                }
            }
        } catch (error) {
            aiText.innerHTML = '<span style="color: #ef4444;">Error fetching search results.</span>';
            console.error("Search Error:", error);
        }
    };

    btnSearch.addEventListener("click", performSearch);
    searchInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") performSearch();
    });


    // ==============================================
    // 4. Task Workspace Logic (NDJSON Streaming)
    // ==============================================
    
    let isAgentRunning = false;
    let currentAbortController = null;

    const addMessage = (content, sender = "user") => {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content agent-response";
        
        if (sender === "system" && typeof marked !== 'undefined') {
            contentDiv.innerHTML = marked.parse(content);
        } else {
            contentDiv.textContent = content;
        }

        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight; 
        return contentDiv; 
    };

    const handleChatSubmit = async () => {
        if (isAgentRunning) {
            if (currentAbortController) currentAbortController.abort();
            resetChatInputState();
            return;
        }

        const prompt = promptInput.value.trim();
        if (!prompt) return;

        addMessage(prompt, "user");
        promptInput.value = "";
        
        isAgentRunning = true;
        actionIcon.className = "fas fa-square";
        actionBtn.classList.add("stop-mode");
        
        const responseNode = addMessage("Initializing Agent...", "system");
        currentAbortController = new AbortController();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: prompt }),
                signal: currentAbortController.signal
            });

            if (!response.ok) throw new Error("Network response was not ok");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let accumulatedMessage = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop(); 
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.type === "tool") {
                            responseNode.innerHTML = `<span style="color:#a1a1aa;"><em><i class="fas fa-cog fa-spin"></i> ${data.name}</em></span>`;
                        } else if (data.type === "msg") {
                            accumulatedMessage += data.content;
                            if (typeof marked !== 'undefined') {
                                responseNode.innerHTML = marked.parse(accumulatedMessage);
                            } else {
                                responseNode.innerText = accumulatedMessage;
                            }
                        } else if (data.type === "error") {
                            responseNode.innerHTML += `<br/><span style="color: #ef4444;">${data.content}</span>`;
                        }
                    } catch (e) {
                        console.warn("Could not parse JSON chunk:", line, e);
                    }
                }
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }

        } catch (error) {
            if (error.name === 'AbortError') {
                responseNode.innerHTML += "<br/><br/><em>[Agent stopped by user]</em>";
            } else {
                responseNode.innerHTML = '<span style="color: #ef4444;">Error communicating with agent.</span>';
                console.error("Chat Error:", error);
            }
        } finally {
            resetChatInputState();
        }
    };

    const resetChatInputState = () => {
        isAgentRunning = false;
        currentAbortController = null;
        actionIcon.className = "fas fa-paper-plane";
        actionBtn.classList.remove("stop-mode");
    };

    actionBtn.addEventListener("click", handleChatSubmit);
    promptInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleChatSubmit();
        }
    });

    promptInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
        if (this.value === "") this.style.height = "auto";
    });


    // ==============================================
    // 5. Search Drawer Settings & Shared Indexing
    // ==============================================

    // 5A. Drawer Open/Close functionality
    const toggleSettingsDrawer = (forceState) => {
        const isOpen = forceState !== undefined ? forceState : !searchSettingsDrawer.classList.contains("open");
        if (isOpen) {
            searchSettingsDrawer.classList.add("open");
            searchSettingsOverlay.classList.add("open");
        } else {
            searchSettingsDrawer.classList.remove("open");
            searchSettingsOverlay.classList.remove("open");
        }
    };

    searchSettingsBtn.addEventListener("click", () => toggleSettingsDrawer(true));
    closeSearchSettingsBtn.addEventListener("click", () => toggleSettingsDrawer(false));
    searchSettingsOverlay.addEventListener("click", () => toggleSettingsDrawer(false));

    // 5B. Syncing Agent Selection state between Task view and Search View
    taskModeSelect.addEventListener("change", (e) => searchModeSelect.value = e.target.value);
    searchModeSelect.addEventListener("change", (e) => taskModeSelect.value = e.target.value);

    // 5C. Shared Reusable Indexing Logic
   const setupIndexer = (inputId, btnId, statusId) => {
        const inputEl = document.getElementById(inputId);
        const btnEl = document.getElementById(btnId);
        const statusEl = document.getElementById(statusId);

        // Hide the old status element so it never takes up empty space/margin
        if (statusEl) statusEl.style.display = "none";

        btnEl.addEventListener("click", async () => {
            const folderPath = inputEl.value.trim();
            const originalBtnText = btnEl.innerHTML;
            const originalPlaceholder = inputEl.getAttribute("placeholder") || "e.g., ./data/docs";

            if (!folderPath) {
                inputEl.classList.add("input-error");
                inputEl.value = "";
                inputEl.placeholder = "Please enter a valid path";
                setTimeout(() => {
                    inputEl.classList.remove("input-error");
                    inputEl.placeholder = originalPlaceholder;
                }, 3000);
                return;
            }

            // Start Loading State (Inline)
            inputEl.classList.add("input-loading");
            inputEl.value = ""; // Clear text so placeholder is visible
            inputEl.placeholder = "Indexing in progress...";
            
            // Swap button text to a spinner
            btnEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            btnEl.disabled = true;

            try {
                const response = await fetch("/api/index", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ folder_path: folderPath }) 
                });

                inputEl.classList.remove("input-loading");

                if (response.ok) {
                    inputEl.classList.add("input-success");
                    inputEl.placeholder = "Indexing complete!";
                    btnEl.innerHTML = '<i class="fas fa-check"></i>';
                    
                    setTimeout(() => {
                        inputEl.classList.remove("input-success");
                        inputEl.placeholder = originalPlaceholder;
                        btnEl.innerHTML = originalBtnText;
                        btnEl.disabled = false;
                    }, 3000);
                } else {
                    throw new Error("Indexing failed");
                }
            } catch (error) {
                inputEl.classList.remove("input-loading");
                inputEl.classList.add("input-error");
                inputEl.placeholder = "Failed to index folder.";
                btnEl.innerHTML = '<i class="fas fa-times"></i>';
                
                setTimeout(() => {
                    inputEl.classList.remove("input-error");
                    inputEl.placeholder = originalPlaceholder;
                    btnEl.innerHTML = originalBtnText;
                    btnEl.disabled = false;
                }, 3000);
            }
        });
    };

    // Apply the indexer functionality to both Task and Search view panels
    setupIndexer("taskFolderInput", "taskIndexBtn", "taskIndexStatus");
    setupIndexer("searchFolderInput", "searchIndexBtn", "searchIndexStatus");

    // ==============================================
    // 6. VNC Monitor Streaming (noVNC)
    // ==============================================
    const vncScreen = document.getElementById('vnc-screen');
    const vncStatus = document.getElementById('vnc-status');
    const vncPlaceholder = document.getElementById('vnc-placeholder');

    // Connect to Websockify (which bridges to your IPv4 5900 port)
    const vncUrl = 'ws://127.0.0.1:6080';

    try {
        const rfb = new RFB(vncScreen, vncUrl);

        rfb.addEventListener("connect", () => {
            console.log("VNC Connected successfully.");
            vncStatus.classList.remove("disconnected");
            vncStatus.classList.add("connected");
            vncPlaceholder.style.display = "none";
        });

        rfb.addEventListener("disconnect", (e) => {
            console.warn("VNC Disconnected.", e);
            vncStatus.classList.remove("connected");
            vncStatus.classList.add("disconnected");
            vncPlaceholder.style.display = "block";
            vncPlaceholder.innerText = "VNC Disconnected. Is Websockify running?";
        });

        // Makes the stream automatically scale to your 16:9 CSS box
        rfb.scaleViewport = true;
        rfb.resizeSession = true;

    } catch (error) {
        console.error("Failed to initialize VNC stream:", error);
        vncPlaceholder.innerText = "Error loading VNC viewer.";
    }

}); // <-- Ensure this remains the end of your DOMContentLoaded wrapper