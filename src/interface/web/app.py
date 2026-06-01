from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import os

# --- Agent Imports ---
from src.core.registry import ServiceRegistry
from src.agents.task_router import route_query
import json

# --- Global Agent Setup ---
registry = ServiceRegistry()

# This lifespan manager ensures your VNC/MCP connections start and stop safely with the server
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up ServiceRegistry...")
    await registry.initialize()
    yield
    print("Shutting down ServiceRegistry...")
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
            "sources": [],
            "iterations": 0,
            "max_iterations": registry.settings.max_iterations
        }
        
        try:
            async with asyncio.timeout(300):  # 5 min timeout
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
                                yield json.dumps({"type": "msg", "content": answer, "sources": sources}) + "\n"

        except asyncio.TimeoutError:
            yield json.dumps({"type": "error", "content": "Search Agent timed out after 5 minutes."}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "content": f"Search Agent Error: {str(e)}"}) + "\n"

    else:
        # Vision agent: perceive → plan → execute → verify loop
        vision_state = {
            "goal": prompt,
            "screenshot_b64": None,
            "action_history": [],
            "step_result": "",
            "done": False,
            "iterations": 0,
            "max_iterations": registry.settings.max_iterations,
            "error": None,
        }

        try:
            async with asyncio.timeout(300):  # 5 min timeout
                async for event in registry.vision_agent.astream(vision_state):
                    for node_name, state_update in event.items():
                        if state_update is None:
                            continue

                        if node_name == "capture_screen":
                            yield json.dumps({"type": "tool", "name": "Capturing screen"}) + "\n"
                            await asyncio.sleep(0.3)
                            yield json.dumps({"type": "tool_done"}) + "\n"

                        elif node_name == "plan_action":
                            history = state_update.get("action_history", [])
                            if history:
                                last = history[-1]
                                label = f"Plan: {last['action_type']} — {last.get('reasoning', '')[:60]}"
                                yield json.dumps({"type": "tool", "name": label}) + "\n"
                                await asyncio.sleep(0.3)
                                yield json.dumps({"type": "tool_done"}) + "\n"

                        elif node_name == "execute_action":
                            step = state_update.get("step_result", "")
                            done = state_update.get("done", False)
                            if done:
                                yield json.dumps({"type": "msg", "content": f"Task complete after {state_update.get('iterations', 0)} steps."}) + "\n"
                            elif step:
                                yield json.dumps({"type": "tool", "name": step}) + "\n"
                                await asyncio.sleep(0.2)
                                yield json.dumps({"type": "tool_done"}) + "\n"

        except asyncio.TimeoutError:
            yield json.dumps({"type": "error", "content": "Vision Agent timed out after 5 minutes."}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "content": f"Vision Agent Error: {str(e)}"}) + "\n"


        
@app.get("/api/config")
async def get_config():
    """Return runtime config for the frontend (VNC URL, etc.)."""
    return {
        "vnc_websocket_port": registry.settings.vnc_websocket_port,
        "vnc_websocket_path": registry.settings.vnc_websocket_path,
    }


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
    if not os.path.isdir(req.folder_path):
        raise HTTPException(status_code=400, detail="Invalid folder path.")
    
    return {"message": f"Not implemented yet"}
    # return {"message": f"Successfully routed {req.folder_path} to indexer."}

if __name__ == "__main__":
    import uvicorn
    # Just run uvicorn. The 'lifespan' hook handles the registry init!
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)