import os
import asyncio
import json

from pathlib import Path
from langchain_core.tools import tool

from src.retrieval.file_loader import AsyncFileLoader
from src.retrieval.web_search import asearch, aretrieve





def get_general_tools(registry=None):

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
        
    @tool
    async def intervention_tool():
        """
        Pauses the agent's execution for 60 seconds to yield control to the human user.
        
        ALWAYS USE THIS TOOL WHEN ENCOUNTERING:
        1. Authentication: login screen, captchas or 2FA.
        2. Stuck State: stuck in a loop, repeating the exact same failed actions for several iterations without making progress."""
        await asyncio.sleep(60)
        return ("[Vision] Resuming Execution")


    @tool
    async def search_agent_tool(query:str, web_search:bool=False)-> str:
        """
        ALWAYS use this tool to retrieve local information or fetch simple information from the web. 
        
        CRITICAL INSTRUCTIONS FOR THE 'query' PARAMETER:
        - Formulate the query as a concise, specific question or a short Google-style keyword search.
        - DO NOT copy and paste your overarching task or give commands to the search agent.
        - Search for a single specific piece of information at a time.
        
        Examples of BAD queries (too long, contains commands):
        - "find local files related to an email sent to Alex or Thunderbird mail data; locate the file path"
        - "search for the bug fix regarding the python recursion error on the website"
        
        Examples of GOOD queries (concise, question/keyword based):
        - "Where is Thunderbird mail data stored?"
        - "What is Alex's emails file path?"
        - "How to fix Python recursion limit error fix?"
        
        Usecases:
        1. local information: retrieve the filepath or content of a file.
        2. web search: find up-to-date information or bug-fixes (set web_search=True).
        """
        if registry is None:
            return "Error: Passed registry is not initialized."
        
        initial_state = {
            "query": query,
            "context_blocks": [],
            "tree_blocks": [], 
            "known_file_paths": [],
            "explored_subtrees": set(),
            "final_answer": "",
            "sources": [],
            "file_summaries": {},
            "web_summaries": {},
            "iterations": 0,
            "max_iterations": registry.settings.max_search_iterations,
            "use_websearch": web_search,
            "needs_websearch_flag": False
        }
        print(f"[Vision] Delegated query: {query}")
        try:
            result = await registry.search_agent.ainvoke(initial_state)
            
            final_answer = result.get("final_answer", "No answer could be generated.")
            sources = result.get("sources", [])
            
            response = f"Search Agent Final Answer:\n{final_answer}\n"
            
            if sources:
                response += "\nSources referenced:\n"
                for src in sources:
                    response += f"- {src.get('name', 'Unknown')} ({src.get('path', 'Unknown Path')})\n"
                    
            return response
            
        except Exception as e:
            return f"Error executing search agent: {str(e)}"
        
    return [
        wait_tool, 
        read_document_tool,
        exploratory_search_tool, 
        read_website_tool, 
        intervention_tool, 
        search_agent_tool
    ]


