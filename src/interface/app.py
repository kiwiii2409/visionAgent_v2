import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.core.registry import ServiceRegistry
from src.interface.stream_handler import generate_chat_stream

registry = ServiceRegistry()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[App] Starting up ServiceRegistry")
    await registry.initialize()
    yield
    print("[App] Shutting down ServiceRegistry...")
    await registry.shutdown()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
TRAY_DIR = os.path.join(BASE_DIR, "tray")

app = FastAPI(lifespan=lifespan)

# Mount frontend-specific static directories
app.mount("/static", StaticFiles(directory=WEB_DIR, html=False), name="static")
app.mount("/tray_static", StaticFiles(directory=TRAY_DIR, html=False), name="tray_static")

class ChatRequest(BaseModel):
    prompt: str

class IndexRequest(BaseModel):
    folder_path: str

@app.get("/api/config")
async def get_config():
    return {
        "vnc_websocket_port": registry.settings.vnc_websocket_port,
        "vnc_websocket_path": registry.settings.vnc_websocket_path,
    }


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

@app.get("/tray")
async def read_tray_index():
    return FileResponse(os.path.join(TRAY_DIR, "tray_index.html"))

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return StreamingResponse(
        generate_chat_stream(req.prompt, registry),
        media_type="text/plain"
    )

@app.post("/api/index")
async def index_endpoint(req: IndexRequest):
    target_path = Path(req.folder_path).resolve()

    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid folder path.")
    
    current_folders = [Path(p).resolve() for p in registry.settings.auto_index_folders]

    # decline indexing if path is subfolder of already indexed directory
    if any(existing == target_path or existing in target_path.parents for existing in current_folders):
        return {"message": "Path is already covered by an indexed parent directory."}
    
    # remove all subdirectories which the new path covers => avoid duplicates
    folders_to_remove = [existing for existing in current_folders if target_path in existing.parents]
    for folder in folders_to_remove:
        registry.settings.auto_index_folders.remove(str(folder))

    registry.settings.auto_index_folders.append(req.folder_path)
    await registry.document_h_indexer.build_index(registry.settings.auto_index_folders)    
    await registry.reload_mcp()
    
    return {"message": f"Successfully added '{req.folder_path}' to index queue."}    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)