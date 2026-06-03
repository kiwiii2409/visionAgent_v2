import asyncio
from langchain_core.tools import tool

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

    return [wait_tool]