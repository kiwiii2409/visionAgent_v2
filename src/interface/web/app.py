from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import os
from pathlib import Path
# --- Agent Imports ---
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver


from src.core.registry import ServiceRegistry
from src.agents.task_router import route_query
import json

# --- Global Agent Setup ---
registry = ServiceRegistry()
memory = MemorySaver()

# This lifespan manager ensures your VNC/MCP connections start and stop safely with the server
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[App] Starting up ServiceRegistry")
    await registry.initialize()
    yield
    print("[App] Shutting down ServiceRegistry...")
    await registry.shutdown()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR, html=False), name="static")

class ChatRequest(BaseModel):
    prompt: str
    # mode: str 

class IndexRequest(BaseModel):
    folder_path: str
async def stream_agent_progress(prompt: str):
    """
    Takes the prompt from the website, feeds it to the LangChain agent,
    and yields JSON strings to construct the UI in real-time.
    """
    task_type = await route_query(prompt, registry.llm)
    yield json.dumps({"type": "init", "mode": task_type}) + "\n"
    
    if task_type == "question":
        initial_state = {
            "query": prompt,
            "context_blocks": [],
            "known_file_paths": [],
            "final_answer": "",
            "sources": []
        }
        
        try:
            async for event in registry.search_agent.astream(initial_state):
                for node_name, state_update in event.items():
                    
                    if state_update is None:
                        continue

                    if node_name == "initial_retrieval":
                        yield json.dumps({"type": "tool", "name": "Retrieving file chunks and maps"}) + "\n"
                        await asyncio.sleep(0.5)
                        yield json.dumps({"type": "tool_done"}) + "\n"
                        
                    elif node_name == "evaluate_context":
                        is_sufficient = state_update.get("is_sufficient_flag")
                        status = "Context sufficient." if is_sufficient else "Missing info. Expanding search..."
                        yield json.dumps({"type": "tool", "name": status}) + "\n"
                        await asyncio.sleep(0.5)
                        yield json.dumps({"type": "tool_done"}) + "\n"
                        
                    elif node_name == "explore_additional_files":
                        files = state_update.get("known_file_paths", [])
                        if files:
                            latest_file = files[-1] 
                            yield json.dumps({"type": "tool", "name": f"Reading file: {latest_file}"}) + "\n"
                            await asyncio.sleep(0.5)
                            yield json.dumps({"type": "tool_done"}) + "\n"
                            
                    elif node_name == "synthesize_answer":
                        answer = state_update.get("final_answer", "")
                        sources = state_update.get("sources", [])
                        if answer:
                            # Added sources array to payload
                            yield json.dumps({"type": "msg", "content": answer, "sources": sources}) + "\n"
                            
        except Exception as e:
            yield json.dumps({"type": "error", "content": f"Search Agent Error: {str(e)}"}) + "\n"

    else:
        # generic agent, switch out for vision agent once done
        mcp_tools = getattr(registry, 'mcp_tools', [])
        all_tools = mcp_tools + registry.retrieval_tools + registry.ui_tools + registry.program_tools
        
        agent = create_agent(registry.llm, all_tools, checkpointer=memory)
        thread_config = {"configurable": {"thread_id": "web_ui_session"}}

        state = agent.get_state(thread_config)
        messages_to_send = []

        if not state.values.get("messages"):
            sys_msg = (
                "You are a specialized local desktop automation agent. "
                "VERY IMPORTANT: You must execute your tools sequentially, ONE AT A TIME. "
                "Never execute multiple mouse or keyboard tools in parallel. "
                "Wait for the outcome of one tool before calling the next."
            )
            messages_to_send.append(SystemMessage(content=sys_msg))

        messages_to_send.append(HumanMessage(content=prompt))
        inputs = {"messages": messages_to_send}

        try:
            async for chunk in agent.astream(inputs, stream_mode="values", config=thread_config):
                message = chunk["messages"][-1]
                
                if message.type == "ai" and message.tool_calls:
                    for tool_call in message.tool_calls:
                        yield json.dumps({"type": "tool", "name": tool_call['name']}) + "\n"
                    await asyncio.sleep(0.5) 
                    
                elif message.type == "tool":
                    yield json.dumps({"type": "tool_done"}) + "\n"
                    
                elif message.type == "ai" and message.content:
                    yield json.dumps({"type": "msg", "content": message.content}) + "\n"
                    
        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"


        
@app.get("/")
async def read_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(index_path)

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return StreamingResponse(
        stream_agent_progress(req.prompt),
        media_type="text/plain"
    )

@app.post("/api/index")
async def index_endpoint(req: IndexRequest):
    target_path = Path(req.folder_path).resolve()

    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid folder path.")
    
    # resolve all tracked folders to check for overlaps
    current_folders = [Path(p).resolve() for p in registry.settings.auto_index_folders]

    # Case target_path folder is part of other
    for existing in current_folders:
        if existing == target_path or existing in target_path.parents:
            return {"message": f"Path is already covered by indexed parent: {existing}"}
    
    folder_to_remove = []
    # Case target_path folder is parent of any existing folders => remove subfolders
    for idx, existing in enumerate(current_folders):
        if target_path in existing.parents:
            folder_to_remove.append(target_path)
    
    for folder in folder_to_remove:
        registry.settings.auto_index_folders.remove(folder)

    # Case target_path folder doesnt overlap (anymore)
    registry.settings.auto_index_folders.append(req.folder_path)

    # rebuild tree from scratch => (saves tokens by reusing unchanged sumamreis)
    await registry.document_h_indexer.build_index(registry.settings.auto_index_folders)    
    
    await registry.reload_mcp()
    return {"message": f"Successfully added '{req.folder_path}' to index queue."}    

if __name__ == "__main__":
    import uvicorn
    # Just run uvicorn. The 'lifespan' hook handles the registry init!
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)