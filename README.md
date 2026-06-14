## How to run:
#### Setup for OmniParser/ YOLO:
<!-- 1. start VPN in separate Terminal: `sudo openvpn --config vpn-air-standard.ovpn` (download config here: https://vpn.ito.cit.tum.de/)  -->
2. start OmniParser or YOLO on Server (run `python3 gradio_demo.py` or `python3 yolo_server.py` in `/data1/visionAgent/OmniParser/`)
3. use port-forwarding, e.g. `ssh -i ~/.ssh/uni_key -L 7861:localhost:7860 manipulation_agent@131.159.60.57`
- new local YOLO option available (weights taken from OmniParser_v2)
  - download the weights:
    - `mkdir -p data/weights/yolo`  
    - `for f in train_args.yaml model.pt model.yaml; do curl -L "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/$f" -o "data/weights yolo/$f"; done`

#### Note: visionAgent operates is currently slowed down on purpose (high wait times before screenshots)

### Mode 1: Agent operates on your screen
1. in `settings.py`: `"local", "tray", "False"`
2. run `python src/main.py`
3. a small desktop tray icon should pop up, double click to open
 
### Mode 2: Agent operates on virtual screen in the backgroudn
1. in `settings.py`: `"virtual", "web", "True"`
2. run `websockify 6080 localhost:5900` in separate terminal
3. run `python src/main.py`
4. visit `http://127.0.0.1:8000/` to monitor the virtual display


## Idea:
- Local website with chat, stream of virtual screen, access to settings/ ability to e.g. add folders to indexed knowledgebase
- two agents implemented as separate graphs, automatic routing of queries to one of both: 
  - **searchAgent**: LLM-based agent, handles pure knowledge retrieval, no manipulation of system ("Google for local system")
  - **visionAgent**: VLM + Omniparser agent, handles all other tasks (writing emails, creating files), interacts with system using tools in a human-like way


## Issues 
- MODE 2: if e.g. vscode is opened on the main display, instructing the agent to "open vscode" will open another instance on the main display
- VisionAgent struggles if yolo doesnt recognize bounding box => currenlty: use yolo bb as primary and predict coordinates as fallback
- "race condition" like behaviour, where sometimes tool-calls with the same goals are executed shortly after each other (slow down kind of fixes it!)
- Occasional random steps/ non-sense tool-calls (e.g. clicking bounding box marking nothing) => maybe try with more capable model (> gpt 5.4 mini) to see whether it's vlm issue


## Already implemented
- **searchAgent** with following flow: 
  - Query -> RAG to get chunks + summaries of surrounding files -> LLM decides: enough information? 
    - if yes: respond
    - if no: select up to 3 surrounding files to fully read and append as context -> respond
- basic tools for retrieval, ui interaction, opening program (currently not used by searchAgent)
- simple web interface to communicate with agent
- hierarchical indexing to allow agent to retrieve additional infromations besides chunks from RAG
- Automatic indexing of only changes in folders using hashing
  - avoids whole reindexing on each startup (`_requires_reindexing`)
  - avoid storing duplicates of the same chunk when reindexing
- Clean up website and agent responses
  - remove emojis from responses
  - make website pretty
  - agent-thinking-bubble in UI doesn't reliably display all steps
- Allow user to change settings on Website
- visionAgent using VLM + Omniparser



## Missing & Ideas
#### Website
- new UI requirement (google-like with AI summary at top and links to documents below)
- 
#### Memory & RAG

#### Agents
- Additional tools for **searchAgent** to apply to context (e.g. count files, get system information, ...)
  - maybe switch similarity search to EnsembleRetriever or Maximal Marginal Relevance (MMR) to fan out retrieved documents (sometimes focus on one folder, but info in anot)
- Memory-features:
  - Shared short-term memory between agents (should we limit that so save tokens?)
  - (maybe) retrieve past (un-) successful high-level task-plans for visionAgent to serve as pos/neg examples
    - requires: user has to be able to "rate" success in the UI after visionAgent finishes a task
- (maybe) additional tools for task execution (closing program, creating/ deleting files) => might require "safe" mode s.t. operations are just logged without being 
- (maybe) Multiple chats, persistent chats
- (maybe) Voice Input, Websearch tool






