import os
import asyncio
import json

from pathlib import Path
from langchain_core.tools import tool

from src.retrieval.file_loader import AsyncFileLoader
from src.retrieval.web_search import asearch, aretrieve



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
    async def exploratory_search_tool(query:str, max_results:int=5) -> str:
        """
        Use this tool FIRST when you need to search the internet for current events, facts, or external knowledge. 
        It performs a web search and returns a list of results containing the title, a brief snippet, and the URL.
        
        Workflow: Evaluate the snippets returned by this tool. If the snippets contain enough information to answer the user, stop here. If you need deeper details, identify the most promising URL from these results and pass it into the `read_website_tool` to read the full page content.
        """
        result = await asearch(query, max_results)
        return result
    
    @tool
    async def read_website_tool(url:str) -> str:
        """
        Use this tool to read the full text content of a specific webpage. 
        
        Workflow: You should typically use this tool AFTER using the `exploratory_websearch_tool`. Do not guess URLs. Take the exact URL/link provided in the search results and pass it into this tool to extract the complete article or webpage data.
        """
        result = await aretrieve(url)
        return result
        


    @tool
    async def read_document_tool(path: str) -> str:
        """
        Reads and extracts text from almost ANY file format.
        Always use this tool to read the contents of a file.
        """
        if not os.path.exists(path):
            return f"Error: File not found at '{path}'"

        try:
            file_loader = AsyncFileLoader(concurreny_limit=3)
            doc = await file_loader.load_single_file(Path(path))
            if doc:
                return doc.page_content
            else:
                return f"Error: File format unsupported or unreadable for '{path}'"
            
        except Exception as e:
            return f"Error reading file '{path}': {str(e)}"
        
    return [wait_tool, read_document_tool,exploratory_search_tool, read_website_tool]