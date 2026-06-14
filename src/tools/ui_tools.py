"""
src/tools/ui_tools.py

Role:
"""
from typing import Literal, Optional
from langchain_core.tools import tool
from src.io.controller import IOController
import asyncio
def get_ui_tools(controller: IOController):
# TODO: once Omniparser works, change method signature to support button number and add translation to coordinates 

    @tool
    async def move_mouse_tool(element_id: Optional[str] = None, x: Optional[float] = None, y: Optional[float] = None) -> str:
        """
        Moves the mouse cursor to a specific UI element without clicking.
        - PRIMARY METHOD: Provide 'element_id' if the destination has a numeric label in the annotated image.
        - FALLBACK METHOD: Only pass 'x' and 'y' (0 to screen width/height) if there is NO bounding box. Estimate the pixel coordinates using nearby labeled boxes.
        Do not provide both element_id and coordinates.
        """
        if x is not None and y is not None:
            await controller.move_mouse(x=x, y=y)
            return f"Successfully moved mouse to ({x}, {y})."
        return "Error: System failed to resolve element_id to coordinates. Please use x,y fallback."

    @tool
    async def click_tool(
        element_id: Optional[str] = None, 
        x: Optional[float] = None, 
        y: Optional[float] = None, 
        button: Literal["left", "right", "middle"] = "left", 
        clicks: int = 1
    ) -> str:
        """
        Clicks a specific UI element on the screen.
        - PRIMARY METHOD: Provide 'element_id' if the destination has a numeric label.
        - FALLBACK METHOD: Only pass 'x' and 'y' directly if there is NO bounding box. Estimate the pixel coordinates using nearby labeled boxes. Text-Fields often lack the bounding-box, be careful there and use direct coordinate prediction if they lack a clear 'element_id'.
        - Use clicks=1 for standard selection. Use clicks=2 to double-click.
        - Set button to "left", "right", or "middle".
        """
        if x is not None and y is not None:
            await controller.move_mouse(x=x, y=y) 
            await asyncio.sleep(0.5)
            await controller.click(button=button, clicks=clicks)
            return f"Successfully moved to and clicked {button} button at ({x}, {y})."
    
        return "Error: System failed to resolve element_id to coordinates. Please use x,y fallback."

    @tool
    async def type_tool(text: str) -> str:
        """
        Types the exact provided string using the keyboard.
        When using for queries try to keep the general and not too specific to avoid specific naming formats or special characters from interfering (e.g. change time -> search for 'time' instead of 'Data & Time')
        CRITICAL: You MUST use the click_tool to click inside a text input field BEFORE using this tool.
        """
        await controller.write(text)
        return f"Successfully typed: '{text}'."


    @tool
    async def key_press_tool(key: str) -> str:
        """
            Presses a specific keyboard key or key combination.
            - Valid single keys: 'enter', 'tab', 'escape', 'space', 'backspace', 'up', 'down'.
            - For shortcuts, use '+' combinations (e.g., 'ctrl+c', 'ctrl+v', 'alt+f4', 'ctrl+s').
            - To open the system menu, use 'win'
            Use this to submit forms ('enter') or paste text ('ctrl+v').
        """
        await controller.key_press(key)
        return f"Successfully pressed key: '{key}'."

    @tool
    async def scroll_tool(amount: int) -> str:
        """
            Scrolls the active window vertically.
            - Use a positive amount (e.g., 500) to scroll UP.
            - Use a negative amount (e.g., -500) to scroll DOWN.
            Note: Ensure you have used move_mouse_tool or click_tool to place the cursor inside the scrollable window first.
        """
        await controller.scroll(amount)
        return f"Successfully scrolled by {amount} units."
    

    # @tool
    # async def drag_and_drop(start_x, start_y, end_x, end_y):
    #     """
    #         Clicks and holds the left mouse button at (start_x, start_y), drags the cursor to (end_x, end_y), and releases.
    #         - Use this for moving windows, dragging scrollbars, or moving files into folders.
    #     """
    #     if (0 <= start_x < screen_width and 0 <= start_y < screen_height and 
    #         0 <= end_x < screen_width and 0 <= end_y < screen_height):
    #         await controller.drag_and_drop(start_x, start_y, end_x, end_y)
    #         return f"Successfully dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})."
    #     else:
    #         return f"Error: One or both coordinates are out of bounds for screen size ({screen_width}, {screen_height})."

    return [move_mouse_tool, click_tool, type_tool, key_press_tool, scroll_tool]