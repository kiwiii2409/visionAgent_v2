# Vision & Search Agent

## General Idea
- Local interface (website or tray) with chat, virtual screen streaming, and system settings (e.g., adding folders to the indexed knowledge base).
- The system features two separate agent graphs with automatic routing of user queries:
  - **searchAgent**: LLM-based agent handling pure knowledge retrieval. Performs no system manipulation ("Google for local system").
  - **visionAgent**: VLM + OmniParser agent that handles tasks (writing emails, creating files) and interacts with the OS in a human-like way via tools.

---

## Configuration & Setup
Settings can be modified in `src/config/settings.py` or overwritten by creating a `.env` file in the project root.

By default, the system uses `gpt-5.4-mini` for LLM/VLM tasks and `BAAI/bge-small-en-v1.5` for embeddings.

### OmniParser / YOLO Setup (Local) (Default)
- A local YOLO option is available using weights from OmniParser_v2. Set `enable_preprocessing = "local"` in settings.
- Download the weights before running:
  - `mkdir -p data/weights/yolo`
  - `for f in train_args.yaml model.pt model.yaml; do curl -L "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/$f" -o "data/weights/yolo/$f"; done`

### OmniParser / YOLO Setup (Uni Server)
- Start OmniParser or YOLO on the server (run `python3 gradio_demo.py` or `python3 yolo_server.py` in `/data1/visionAgent/OmniParser/`).
- Use port-forwarding to connect locally, e.g.:
  `ssh -i ~/.ssh/uni_key -L 7861:localhost:7860 manipulation_agent@131.159.60.57`
- Set `enable_preprocessing = "server"` in settings.

### Local Search / LLM Setup (Optional)
- You can enable local models instead of OpenAI by setting `enable_local_search = True` in your `.env` or settings.
- The system defaults to Ollama running at `http://127.0.0.1:11434` with the `qwen3.5:4b` model.

---

## How to Run (Modes)
Task and Search Agent are supported in both modes. Mode 1 offers separate interfaces for search and task, while Mode 2 offers a shared interface with automatic routing.


### Mode 1: Agent operates on a virtual screen in the background
1. In `settings.py` or `.env`, set the following:
   - `display_mode = "virtual"`
   - `ui_mode = "web"`
   - `enable_vnc = True`
2. Run `websockify 6080 localhost:5900` in a separate terminal. This allows you to monitor the virtual display.
3. Run `python src/main.py`.
4. Visit `http://127.0.0.1:8000/` in your browser to chat and monitor the virtual display.

### Mode 2: Agent operates on your main screen
1. In `settings.py` or `.env`, set the following:
   - `display_mode = "local"`
   - `ui_mode = "tray"`
   - `enable_vnc = False`
2. Run `python src/main.py`.
3. A small desktop tray icon should pop up. Double-click it to open the interface.



---

## Current Issues 
- **YOLO Bounding Boxes:** visionAgent struggles if YOLO doesn't recognize a bounding box. 
  - **Solution**: Use YOLO bounding boxes as primary and predict coordinates as a fallback.*
- **Race Conditions:** Tool-calls with the same goals are sometimes executed too quickly in succession. 
  - **Solution:** The intentional slowdown currently acts as a temporary fix
- **Hallucinated Actions:** Occasional random steps or nonsense tool-calls occur (e.g., clicking a bounding box that marks nothing)
- **Virtual Screen bleed:** In Mode 2, if an app like VSCode is already open on the main display, instructing the agent to "open vscode" will open another instance on the main display rather than the virtual one.
---

## Missing & Ideas

#### Agents & Tools
- **searchAgent:**
  - Improve context filtering to increase token efficiency.

- **visionAgent:**
  - Improve reliability and speed.

- **General:**
  - Voice input.

#### Memory & RAG
- Retrieve past (un)successful high-level task-plans for visionAgent to serve as positive/negative examples (requires a UI feature for users to "rate" task success).

