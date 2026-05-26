## How to run:
1. activate virtual environment and install requirements: `pip install -e .`
2. set the folder you want to index in the `.env` 
   - **Note:** for testing choose a small folder (e.g. this codebase) to avoid wasting api credits and speedup the first startup.
3. `python src/interface/web/app.py`
4. in a separate terminal to stream vnc to browser: `websockify 6080 localhost:5900`
5. visit: `http://127.0.0.1:8000/` and type request (e.g. open chrome or ask about context of file in indexed folder)

**Note:** the visionAgent isn't implemented yet, therefore a ready-to-use langchain agent is used as replacement. Basic tool usage is supported but the agent doesn't get visual feedback and can therefore only follow instructions (e.g. move mouse to 100,100; open thunderbird)

## Idea:
- Local website with chat, stream of virtual screen, access to settings/ ability to e.g. add folders to indexed knowledgebase
- two agents implemented as separate graphs, automatic routing of queries to one of both: 
  - **searchAgent**: LLM-based agent, handles pure knowledge retrieval, no manipulation of system ("Google for local system")
  - **visionAgent**: VLM + Omniparser agent, handles all other tasks (writing emails, creating files), interacts with system using tools in a human-like way


## Already implemented
- **searchAgent** with following flow: 
  - Query -> RAG to get chunks + summaries of surrounding files -> LLM decides: enough information? 
    - if yes: respond
    - if no: select up to 3 surrounding files to fully read and append as context -> respond
- basic tools for retrieval, ui interaction, opening program (currently not used by searchAgent)
- simple web interface to communicate with agent
- hierarchical indexing to allow agent to retrieve additional infromations besides chunks from RAG

## Missing & Ideas
#### Website
- Clean up website and agent responses
  - remove emojis from responses
  - make website pretty
  - agent-thinking-bubble in UI doesn't reliably display all steps
- Allow user to add folders to index or change settings on Website

#### Memory & RAG
- Automatic indexing of only changes in folders using hashing + last-changed date (?)
  - avoids whole reindexing on each startup (`_requires_reindexing`)
  - avoid storing duplicates of the same chunk when reindexing

#### Agents
- Additional tools for **searchAgent** to apply to context (e.g. count files, get system information, ...)
- Memory-features:
  - Shared short-term memory between agents (should we limit that so save tokens?)
  - (maybe, low priority) retrieve past (un-) successful high-level task-plans for visionAgent to serve as pos/neg examples
    - requires: user has to be able to "rate" success in the UI after visionAgent finishes a task
- (low priority) additional tools for task execution (closing program, creating/ deleting files) => might require "safe" mode s.t. operations are just logged without being 
- (low priority) visionAgent using VLM + Omniparser
- (low priority) Multiple chats, persistent chats

## Issues 
- if e.g. vscode is opened on the main display, instructing the agent to "open vscode" will open another instance on the main display





