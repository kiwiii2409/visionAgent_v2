import os
import asyncio
from pathlib import Path
from langchain_core.tools import tool

from src.retrieval.file_loader import AsyncFileLoader

def get_general_tools():

    @tool
    async def wait_tool(seconds: float) -> str:
        """
        Pauses graph execution to wait for the screen, application, or network to load.
        Use this tool after actions that require rendering time (e.g., launching an app, clicking a hyperlink, submitting a form).
        
        Guidelines for 'seconds':
        - 1.0 to 2.0: Quick UI updates, opening local menus, or typing in text fields.
        - 3.0 to 5.0: Loading standard web pages or opening lightweight applications.
        - 5.0 to 10.0: Launching heavy applications (like Thunderbird or Firefox) or waiting for large downloads.
        
        Note: The wait time is hard-capped at 10.0 seconds per call.
        """
        safe_seconds = min(float(seconds), 10.0) 
        await asyncio.sleep(safe_seconds)
        return f"Successfully waited for {safe_seconds} seconds."




    @tool
    async def read_document_tool(path: str) -> str:
        """
        Reads and extracts text from almost ANY file format.
        Always use this tool to read the contents of a file.
        """
        if not os.path.exists(path):
            return f"Error: File not found at '{path}'"

        try:
            file_loader = AsyncFileLoader(concurrency_limit=5)
            doc = await file_loader.load_single_file(Path(path))
            if doc:
                return doc.page_content
            else:
                return f"Error: File format unsupported or unreadable for '{path}'"
            
        except Exception as e:
            return f"Error reading file '{path}': {str(e)}"
        
    return [wait_tool, read_document_tool]