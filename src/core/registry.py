"""
src/core/registry.py

Role:
    register all servies etc. at a single point and initialize the system
"""

import os
import json
import hashlib
import subprocess
from pyvirtualdisplay import Display
from typing import List, Any
from pathlib import Path

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

# tools
from src.tools.ui_tools import get_ui_tools
from src.tools.retrieval_tools import get_retrieval_tools
from src.tools.program_tools import get_program_tools


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

        # init screen, must be done before importing pyautogui to set correct display
        self._init_virtual_display()


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

        # vector store — suppress HF warnings and progress bars
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=f"{self.settings.embedding_model}"
        )
        self.vector_store = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.settings.chroma_db_path
        )

        # init services
        self.controller = IOController()
        self.screen_capture = ScreenCapture()
        # self.document_indexer = DocumentIndexer(
        #     vector_store=self.vector_store,
        #     chunk_size=self.settings.chunk_size,
        #     chunk_overlap=self.settings.chunk_overlap
        # )
        tree_file = Path(self.settings.summary_tree_path) / self.settings.summary_tree_filename
        self.document_h_indexer = HierarchicalIndexer(
            llm=self.llm,
            vector_store=self.vector_store,
            summary_tree_path=str(tree_file),
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )

        # gather tools
        self.ui_tools = get_ui_tools(
            self.controller, self.settings.virtual_resolution[0], self.settings.virtual_resolution[1])
        self.retrieval_tools = get_retrieval_tools(self.vector_store)
        self.program_tools = get_program_tools()


        self._initialized = True

    async def initialize(self) -> None:
        """ does the hierarchical indexing, mcp init and agent init"""
        folders_to_index = self._requires_reindexing()
        if folders_to_index:
            await self.document_h_indexer.build_index(folders_to_index)
            print(f"[Registry] Successfully indexed {len(folders_to_index)} folders")
            # Save file hashes for future incremental indexing
            self._save_file_hashes(folders_to_index)
        else:
            print("[Registry] Existing index is up to date. Booting instantly.")

        await self._init_mcp()

        mcp_tools_dict = {tool.name: tool for tool in self.mcp_tools}

        search_builder = SearchGraphBuilder(
            llm=self.llm,
            vectorstore=self.vector_store,
            mcp_tools_dict=mcp_tools_dict,
            summary_tree_path=str(tree_file),
            max_iterations=self.settings.max_iterations,
            retrieval_k=self.settings.retrieval_top_k,
        )

        self.search_agent = search_builder.build()

        # Vision agent: perceive → plan → execute → verify loop
        vision_builder = VisionGraphBuilder(
            vlm=self.vlm,
            screen_capture=self.screen_capture,
            io_controller=self.controller,
            max_iterations=self.settings.max_iterations,
        )
        self.vision_agent = vision_builder.build()
        print("[Registry] Vision agent built")

    def _init_virtual_display(self) -> None:
        if self.settings.docker_mode:
            print(f"[Registry] Docker mode: using container X11 display at DISPLAY={os.environ.get('DISPLAY', ':1')}")
            return
        if not self.settings.use_virtual_display:
            return

        print("[Registry] Starting Virtual Display")
        self.display = Display(
            visible=0, size=self.settings.virtual_resolution)
        self.display.start()

        print(
            f"[Registry] Agent Active on DISPLAY={self.display.new_display_var}")

        if getattr(self.settings, 'enable_vnc', False):
            print(
                f"[Registry] Starting VNC Server on port {self.settings.vnc_port}")
            # Clear Wayland env vars - x11vnc 0.9.16 refuses to start on Wayland
            vnc_env = os.environ.copy()
            vnc_env.pop("WAYLAND_DISPLAY", None)
            vnc_env.pop("XDG_SESSION_TYPE", None)
            self.vnc_process = subprocess.Popen([
                "x11vnc",
                "-display", self.display.new_display_var,
                "-nopw",
                "-listen", "0.0.0.0",
                "-rfbport", str(self.settings.vnc_port),
                "-forever",
                "-quiet",
                "-cursor", "arrow"
            ], env=vnc_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Start a lightweight window manager on the virtual display.
        # Without this the screen stays black and GUI apps fail to render.
        self._start_window_manager()

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

    # async def _index_folders(self) -> None:
    #     sum_files = 0
    #     sum_chunks = 0
    #     for folder in self.settings.auto_index_folders:
    #         try:
    #             result = await self.document_indexer.index_path(folder)
    #             sum_files += result.get('files_scanned')
    #             sum_chunks += result.get('chunks_indexed')

    #         except Exception as e:
    #             print(f"     Error indexing '{folder}': {e}")
    #     print(
    #         f"[Registry] Indexed {sum_chunks} chunks from {sum_files} files.")

    def _save_file_hashes(self, folders: List[str]) -> None:
        """Save SHA256 hashes of all indexed files for incremental reindex detection."""
        hashes = {}
        for folder in folders:
            root = Path(folder)
            if not root.exists():
                continue
            for file_path in root.rglob("*"):
                if file_path.is_file() and not any(
                    p in file_path.parts for p in (".git", "__pycache__", ".egg-info")
                ):
                    try:
                        hashes[str(file_path)] = hashlib.sha256(file_path.read_bytes()).hexdigest()
                    except (IOError, PermissionError):
                        continue
        hashes_path = Path(self.settings.summary_tree_path) / "file_hashes.json"
        with open(hashes_path, "w") as f:
            json.dump(hashes, f)
        print(f"[Registry] Saved {len(hashes)} file hashes for incremental indexing")

    def _requires_reindexing(self) -> List[str]:
        """Check which indexed folders have new, changed, or deleted files.

        Computes SHA256 hashes of all files in auto_index_folders,
        compares against stored hashes from the last successful index.
        Returns folders that need re-indexing (or empty list if up to date).
        """
        if not self.settings.auto_index_folders:
            return []

        hashes_path = Path(self.settings.summary_tree_path) / "file_hashes.json"
        tree_path = Path(self.settings.summary_tree_path) / self.settings.summary_tree_filename

        # First run: no stored state
        if not tree_path.exists() or not hashes_path.exists():
            return self.settings.auto_index_folders

        # Load stored hashes
        try:
            with open(hashes_path, "r") as f:
                stored_hashes = json.load(f)
        except (json.JSONDecodeError, IOError):
            return self.settings.auto_index_folders

        # Compute current hashes for all files in indexed folders
        current_hashes = {}
        for folder in self.settings.auto_index_folders:
            root = Path(folder)
            if not root.exists():
                continue
            for file_path in root.rglob("*"):
                if file_path.is_file() and not any(
                    p in file_path.parts for p in (".git", "__pycache__", ".egg-info")
                ):
                    try:
                        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
                        current_hashes[str(file_path)] = digest
                    except (IOError, PermissionError):
                        continue

        # Detect changes
        stored_set = set(stored_hashes.keys())
        current_set = set(current_hashes.keys())

        new_or_changed = {
            f for f in current_set
            if f not in stored_set or current_hashes[f] != stored_hashes[f]
        }
        deleted = stored_set - current_set

        if not new_or_changed and not deleted:
            return []

        print(f"[Registry] Detected {len(new_or_changed)} new/changed files, "
              f"{len(deleted)} deleted files — reindexing")
        return self.settings.auto_index_folders

    async def shutdown(self) -> None:
        print("[Registry] Shutting down")

        if hasattr(self, 'vnc_process'):
            self.vnc_process.terminate()

        if hasattr(self, 'wm_process'):
            self.wm_process.terminate()

        # Only stop the display if we started it (not in Docker mode)
        if hasattr(self, 'display') and not self.settings.docker_mode:
            self.display.stop()
