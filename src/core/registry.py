"""
src/core/registry.py

Role:
    register all servies etc. at a single point and initialize the system
"""

import subprocess
from pyvirtualdisplay import Display
from typing import List, Any

# langchain 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
# mcp
from langchain_mcp_adapters.client import MultiServerMCPClient


from src.config.settings import Settings, PROJECT_ROOT
from src.io.controller import IOController
from src.io.capture import ScreenCapture
from src.retrieval.indexer import DocumentIndexer

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

        # vector store 
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
        self.document_indexer = DocumentIndexer(
            vector_store=self.vector_store,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )


        self.llm = ChatOpenAI(
            model=self.settings.llm_model_name,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.api_base_url
        )

        # gather tools
        self.ui_tools = get_ui_tools(self.controller, self.settings.virtual_resolution[0], self.settings.virtual_resolution[1])
        self.retrieval_tools = get_retrieval_tools(self.vector_store)
        self.program_tools = get_program_tools()
        
        self._initialized = True



    async def initialize(self) -> None:
            await self._init_mcp()
            await self._index_folders()

    def _init_virtual_display(self) -> None:
        if not self.settings.use_virtual_display:
            return
            
        print("[Registry] Starting Virtual Display")
        self.display = Display(visible=0, size=self.settings.virtual_resolution)
        self.display.start()
        
        print(f"[Registry] Agent Active on DISPLAY={self.display.new_display_var}")
        
        if getattr(self.settings, 'enable_vnc', False):
            print(f"[Registry] Starting VNC Server on port {self.settings.vnc_port}")
            self.vnc_process = subprocess.Popen([
                "x11vnc", 
                "-display", self.display.new_display_var, 
                "-nopw", 
                "-listen", "0.0.0.0", 
                "-rfbport", str(self.settings.vnc_port),
                "-forever", 
                "-quiet",
                "-cursor", "arrow" 
            ])

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
        print(f"[Registry] Successfully loaded {len(self.mcp_tools)} MCP tools.")
    
    async def _index_folders(self) -> None:
        sum_files = 0
        sum_chunks = 0
        for folder in self.settings.auto_index_folders:
            try:
                result = await self.document_indexer.index_path(folder)
                sum_files += result.get('files_scanned')
                sum_chunks += result.get('chunks_indexed')

            except Exception as e:
                print(f"     Error indexing '{folder}': {e}")
        print(f"[Registry] Indexed {sum_chunks} chunks from {sum_files} files.")
    

    async def shutdown(self) -> None:
        print("[Registry] Shutting down")

            
        if hasattr(self, 'vnc_process'):
            self.vnc_process.terminate()
            
        if hasattr(self, 'display'):
            self.display.stop()
            