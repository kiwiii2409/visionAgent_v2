"""
src/tools/ui_tools.py

Role:
"""
from typing import Literal
from langchain_core.tools import tool
from src.io.controller import IOController

def get_ui_tools(controller: IOController, screen_width: int = 1920, screen_height: int = 1080):
    
    @tool
    async def move_mouse_tool(x: float, y: float) -> str:
        """Move the mouse cursor to absolute coordinates, top Left being (0,0)"""
        if x < screen_width and x >= 0 and y < screen_height and y >= 0:
            await controller.move_mouse(x=x, y=y)
            return f"Successfully moved mouse to ({x}, {y})."
        else:
            return f"Error: Coordinates ({x}, {y}) are out of bounds for the screen size ({screen_width}, {screen_height})."

    @tool
    async def click_tool(x: float, y: float, button: Literal["left", "right", "middle"] = "left") -> str:
        """Click at specified coordinates or current position."""
        if x < screen_width and x >= 0 and y < screen_height and y >= 0:
            await controller.click(x=x, y=y, button=button)
            return f"Successfully clicked {button} button at ({x}, {y})."
        else:
            return f"Error: Coordinates ({x}, {y}) are out of bounds for the screen size ({screen_width}, {screen_height})."

    @tool
    async def type_tool(text: str) -> str:
        """Type specifified text"""
        await controller.write(text)
        return f"Successfully typed: '{text}'."

    @tool
    async def key_press_tool(key: str) -> str:
        """Press the specified key (e.g., 'enter', 'tab', 'ctrl')"""
        await controller.key_press(key)
        return f"Successfully pressed key: '{key}'."

    @tool
    async def scroll_tool(amount: int) -> str:
        """Scroll up (positive) or down (negative) at given coordinates"""
        await controller.scroll(amount)
        return f"Successfully scrolled by {amount} units."

    return [move_mouse_tool, click_tool, type_tool, key_press_tool, scroll_tool]