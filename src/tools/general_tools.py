import os
import asyncio
from langchain_core.tools import tool
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_unstructured import UnstructuredLoader


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
        Supported formats: .pdf, .docx, .pptx, .xlsx, .csv, .html, .md, .txt, .py, etc.
        Always use this tool to read the contents of a file.
        """
        if not os.path.exists(path):
            return f"Error: File not found at '{path}'"

        try:
            # apparently pymupdf is much faster than unstrucutredloader for pdfs
            if path.lower().endswith('.pdf'):
                loader = PyMuPDFLoader(path)
                
            else:
                loader = UnstructuredLoader(file_path=path)
                
            docs = await asyncio.to_thread(loader.load)
            
            full_text = "\n\n".join([doc.page_content for doc in docs])
            
            return full_text
            
        except Exception as e:
            return f"Error reading file '{path}': {str(e)}"
        
    return [wait_tool, read_document_tool]