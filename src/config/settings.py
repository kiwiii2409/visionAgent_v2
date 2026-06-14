from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Tuple, Literal
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
    vlm_model_name: str = "gpt-5.4-mini"
    llm_model_name: str = "gpt-5.4-nano"
    embedding_model: str = "BAAI/bge-small-en-v1.5"


    enable_preprocessing: bool = True
    preprocessing_base_url: str = "http://127.0.0.1:8020"

    

    # common configs:
    # local, tray, enable_vnc = false
    # virtual, web, enable_vnc = true
    display_mode: Literal["local", "virtual", "docker"] = "local"
    ui_mode: Literal["tray", "web"] = "tray"
    virtual_resolution: Tuple[int, int] = (1920, 1080)
    
    enable_vnc: bool = False 
    vnc_port: int = 5900
    vnc_websocket_port: int = 6080
    vnc_websocket_path: str = "/"

    
    chroma_db_path: str = str(PROJECT_ROOT / "data" / "chroma")    
    collection_name: str = "visionAgentDocs"
    indexing_path: str = str(PROJECT_ROOT / "data" / "indexing")
    summary_tree_filename: str = "summary_tree.json"
    file_hashes_filename: str = "file_hashes.json"


    retrieval_top_k: int = 4
    chunk_size: int = 512
    chunk_overlap: int = 128
    auto_index_folders: List[str] = [] # gets overwritten by .env value 
    max_iterations: int = 30
    max_search_iterations: int = 2
