from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Tuple
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):    
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / '.env'), 
        env_file_encoding='utf-8', 
        extra='ignore' # .env takes priority and overwrites values specified in settings.py
    )

    openai_api_key: str = "dummy-key" # gets overwritten by .env value
    api_base_url: str = "https://api.openai.com/v1"

    use_virtual_display: bool = True
    virtual_resolution: Tuple = (1920,1080)
    enable_vnc: bool = True
    vnc_port: int = 5900

    vlm_model_name: str = "gpt-5.4-mini"
    llm_model_name: str = "gpt-5.4-nano"
    
    chroma_db_path: str = str(PROJECT_ROOT / "data" / "chroma")    
    collection_name: str = "visionAgentDocs"

    # location of summary tree, which contains summaries of all files for the agent to see
    indexing_path: str = str(PROJECT_ROOT / "data" / "indexing")
    summary_tree_filename: str = "summary_tree.json"
    file_hashes_filename: str = "file_hashes.json"

    retrieval_top_k: int = 4

    # VNC / WebSocket config for the browser-side noVNC viewer
    vnc_websocket_port: int = 6080
    vnc_websocket_path: str = "/"

    # Docker / container mode
    # When True, the agent uses the container's existing X11 desktop
    # instead of starting its own Xvfb + x11vnc.
    docker_mode: bool = False

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 128
    
    auto_index_folders: List[str] = [] # gets overwritten by .env value 
    
    max_iterations: int = 15
    max_search_iterations: int = 2