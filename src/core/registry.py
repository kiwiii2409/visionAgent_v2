"""
src/core/registry.py

Role:
    register all servies etc. at a single point and initialize the system
"""

import os
import json
import hashlib
import subprocess
import logging
from pyvirtualdisplay import Display
from typing import List, Any
from pathlib import Path
import asyncio

# Suppress noisy startup output — must run before HuggingFace imports
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)

# langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# mcp
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config.settings import Settings, PROJECT_ROOT
from src.io.controller import IOController
from src.io.capture import ScreenCapture
from src.retrieval.hierarchical_indexer import HierarchicalIndexer

from src.agents.search_graph import SearchGraphBuilder
from src.agents.vision_graph import VisionGraphBuilder

from src.io.vision.yolo_client import AsyncYoloClient

# tools
from src.tools.ui_tools import get_ui_tools
from src.tools.retrieval_tools import get_retrieval_tools
from src.tools.program_tools import get_program_tools
from src.tools.general_tools import get_general_tools



class ServiceRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        
        self.settings = Settings()
        self._setup_display()
        self._init_models()
        self._init_services()
        self.all_tools = get_ui_tools(self.controller, self.settings.virtual_resolution[0], self.settings.virtual_resolution[1]) + get_general_tools() #+ get_program_tools() # TODO: activate get_program_tool for testing if UI opening doesnt owrk

        self._initialized = True

        


    async def initialize(self) -> None:
        """After main init due to async, does hierarchical indexing, mcp init and agent init"""
        if getattr(self, '_async_initialized', False):
            return
        self._async_initialized = True

        if self.settings.auto_index_folders:
            await self.document_h_indexer.build_index(self.settings.auto_index_folders)
            print(f"[Registry] Successfully indexed {len(self.settings.auto_index_folders)} folders")
        self.indexing_task = asyncio.create_task(self._background_indexer(interval_minutes=5)) # runs automatic indexing every 5 mins

        await self._init_mcp()
        self.all_tools += self.mcp_tools
        self._build_agents()
        self._initialized = True
        
        print("[Registry] Indexing started, MCP Tools and Agents initialized")


# display related methods:
    def _setup_display(self) -> None:
        """handles screen mapping (Local, Virtual, Docker) and VNC"""
        mode = self.settings.display_mode

        if mode == "docker":
            print(f"[Registry] Docker mode: using container X11 display at DISPLAY={os.environ.get('DISPLAY', ':1')}")
        
        elif mode == "local":
            print(f"[Registry] Local mode: controlling host display at DISPLAY={os.environ.get('DISPLAY', ':0')}")
        
        elif mode == "virtual":
            print("[Registry] Starting Virtual Display")
            self.display = Display(visible=0, size=self.settings.virtual_resolution)
            self.display.start()
            print(f"[Registry] Agent Active on DISPLAY={self.display.new_display_var}")
            self._start_window_manager()
        
        # Start VNC server regardless of mode if requested (hooks into current DISPLAY)
        if self.settings.enable_vnc:
            self._start_vnc()

    def _start_vnc(self) -> None:
        """Start the x11vnc server pointing to the currently active DISPLAY"""
        display_var = os.environ.get("DISPLAY", getattr(self, 'display', None) and self.display.new_display_var)
        print(
            f"[Registry] Starting VNC Server on port {self.settings.vnc_port}")
        # Clear Wayland env vars - x11vnc 0.9.16 refuses to start on Wayland
        vnc_env = os.environ.copy()
        vnc_env.pop("WAYLAND_DISPLAY", None)
        vnc_env.pop("XDG_SESSION_TYPE", None)
        self.vnc_process = subprocess.Popen([
            "x11vnc",
            "-display", display_var,
            "-nopw",
            "-listen", "0.0.0.0",
            "-rfbport", str(self.settings.vnc_port),
            "-forever",
            "-quiet",
            "-cursor", "arrow"
        ], env=vnc_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)




    def _start_window_manager(self) -> None:
        """Launch a lightweight window manager on the virtual display.

        Without a WM the Xvfb screen stays black and GUI apps (Firefox, etc.)
        cannot render. Tries fluxbox first, then openbox, then gives up.
        """
        display_var = os.environ.get("DISPLAY", self.display.new_display_var)
        wm_candidates = ["fluxbox", "openbox", "icewm", "jwm"]

        for wm in wm_candidates:
            wm_path = subprocess.run(["which", wm], capture_output=True, text=True)
            if wm_path.returncode == 0:
                print(f"[Registry] Starting window manager: {wm} on DISPLAY={display_var}")
                wm_env = os.environ.copy()
                wm_env["DISPLAY"] = display_var
                self.wm_process = subprocess.Popen(
                    [wm_path.stdout.strip()],
                    env=wm_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return

        print("[Registry] WARNING: No window manager found (fluxbox/openbox). "
              "Install with: sudo apt-get install -y fluxbox")


    def _init_models(self) -> None:
        """Init VLM, LLM, Chroma + Embeddings"""
        self.llm = ChatOpenAI(
            model=self.settings.llm_model_name,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.api_base_url
        )

        # VLM for vision-based agent (supports image inputs)
        self.vlm = ChatOpenAI(
            model=self.settings.vlm_model_name,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.api_base_url,
            max_tokens=1024,
        )

        self.preprocessor = None
        if self.settings.enable_preprocessing:
            self.preprocessor = AsyncYoloClient(self.settings.preprocessing_base_url)



        # vector store — disable tqdm briefly during model load
        _tqdm_disabled = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=f"{self.settings.embedding_model}"
        )
        if _tqdm_disabled is None:
            del os.environ["TQDM_DISABLE"]
        else:
            os.environ["TQDM_DISABLE"] = _tqdm_disabled
        self.vector_store = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.settings.chroma_db_path
        )

    def _init_services(self) -> None:
        """Register all controllers + the indexer"""
        self.controller = IOController()
        self.screen_capture = ScreenCapture()

        self.summary_tree_path = Path(self.settings.indexing_path) / self.settings.summary_tree_filename
        self.file_hashes_path = Path(self.settings.indexing_path) / self.settings.file_hashes_filename

        self.document_h_indexer = HierarchicalIndexer(
            llm=self.llm,
            vector_store=self.vector_store,
            summary_tree_path=str(self.summary_tree_path),
            file_hashes_path = str(self.file_hashes_path),
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )


# called separately due to async issues wtih __init__
    async def _init_mcp(self) -> None:
        print("[Registry] Starting local MCP server for filesystem")
        self.mcp_client = MultiServerMCPClient({
            "local_filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", *self.settings.auto_index_folders]
            }
        })

        self.mcp_tools = await self.mcp_client.get_tools()
        print(
            f"[Registry] Successfully loaded {len(self.mcp_tools)} MCP tools.")
   
    def _build_agents(self) -> None:
        self.search_builder = SearchGraphBuilder(
            llm=self.llm,
            vectorstore=self.vector_store,
            mcp_tools=self.all_tools,
            summary_tree_path=str(self.summary_tree_path),
            max_iterations=self.settings.max_iterations,
            retrieval_k=self.settings.retrieval_top_k,
        )

        self.search_agent = self.search_builder.build()

        # Vision agent: perceive → plan → execute → verify loop
        self.vision_builder = VisionGraphBuilder(
            vlm=self.vlm,
            mcp_tools=self.all_tools,
            screen_capture=self.screen_capture,
            preprocessor=self.preprocessor,
            max_iterations=self.settings.max_iterations,
        )
        self.vision_agent = self.vision_builder.build()




    async def reload_mcp(self) -> None:
        """restart the mcp server to update tool permission, used after e.g. new file path was added """
        print("\n[Registry] Restarting MCP server to update tool permissions (currently ONLY searchAgent)")
        
        if hasattr(self, 'mcp_client') and self.mcp_client is not None:
            print("[Registry] Shutting down existing MCP")
            try:
                await self.mcp_client.disconnect() 
            except Exception as e:
                print(f"[Registry] Note: Error during MCP shutdown: {e}")

        await self._init_mcp()
        
        new_tools_dict = {tool.name: tool for tool in self.all_tools}
        self.search_builder.mcp_tools_dict = new_tools_dict
        self.vision_builder.mcp_tools_dict = new_tools_dict
        
        print("[Registry] Restarted MCP and updated tools successfully")

    async def _background_indexer(self,interval_minutes):
        while True:
            await asyncio.sleep(interval_minutes * 60)
            if self.settings.auto_index_folders:
                try:
                    await self.document_h_indexer.build_index(self.settings.auto_index_folders)
                except Exception as e:
                    print(f"[Registry] Background indexer encountered an error: {e}") 
    
    async def shutdown(self) -> None:
        print("[Registry] Shutting down")

        if hasattr(self, 'mcp_client') and self.mcp_client is not None:
            try:
                await self.mcp_client.disconnect()
            except Exception:
                pass
        
        if hasattr(self, 'indexing_task'):
            self.indexing_task.cancel()

        if hasattr(self, 'vnc_process'):
            self.vnc_process.terminate()
            try:
                self.vnc_process.wait(timeout=1.0) 
            except subprocess.TimeoutExpired:
                self.vnc_process.kill()

        if hasattr(self, 'wm_process'):
            self.wm_process.terminate()
            try:
                self.wm_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.wm_process.kill()

        # Only stop the display if we started it 
        if getattr(self, 'display', None) and self.settings.display_mode == "virtual":
            self.display.stop()
