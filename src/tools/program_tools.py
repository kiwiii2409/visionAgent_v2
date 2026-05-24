"""
src/tools/program_tools.py

Role:
    tools to open programs, run terminal commands etc.
"""
import os
import subprocess 
from langchain_core.tools import tool
from typing import Literal

def get_program_tools():

    @tool
    async def launch_application_tool(app_command: str) -> str:
        """
        lauch desktop applications by their name e.g. 'firefox', 'thunderbird'
        """
        try:
            env = os.environ.copy()
            
            subprocess.Popen(
                app_command,
                env=env,
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            return f"Successfully launched '{app_command}'."
            
        except FileNotFoundError:
            return f"Error: The application '{app_command}' is not installed or not in the system PATH."
        except Exception as e:
            return f"Error launching '{app_command}': {str(e)}"
        
    return [launch_application_tool]