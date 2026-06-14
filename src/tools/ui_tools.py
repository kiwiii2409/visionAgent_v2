"""
src/tools/ui_tools.py

Role:
"""
from typing import Literal
from langchain_core.tools import tool
from src.io.controller import IOController

def get_ui_tools(controller: IOController, screen_width: int = 1920, screen_height: int = 1080):
# TODO: once Omniparser works, change method signature to support button number and add translation to coordinates 

    @tool
    async def move_mouse_tool(x: float, y: float) -> str:
        """
            Moves the mouse cursor to a specific UI element without clicking.
            CRITICAL INSTRUCTION: You MUST provide 'element_id' (the numeric label from the image) in your tool_args instead of x and y. The system will auto-translate it.
            - Use this to hover over elements to reveal tooltips, expand dropdown menus, or prepare for scrolling.
            - Fallback: Only pass 'x' and 'y' directly if there is no bounding box and you must guess the pixel location.
        """
        if x < screen_width and x >= 0 and y < screen_height and y >= 0:
            await controller.move_mouse(x=x, y=y)
            return f"Successfully moved mouse to ({x}, {y})."
        else:
            return f"Error: Coordinates ({x}, {y}) are out of bounds for the screen size ({screen_width}, {screen_height})."

    @tool
    async def click_tool(x: float, y: float, button: Literal["left", "right", "middle"] = "left", clicks: int = 1) -> str:
        """
            Clicks a specific UI element on the screen.
            CRITICAL INSTRUCTION: You MUST provide 'element_id' (the numeric label from the image) in your tool_args instead of x and y. The system will auto-translate it.
            - Use clicks=1 for standard selection, clicking buttons, or focusing text input fields.
            - Use clicks=2 to double-click (e.g., opening applications from the desktop, selecting a full word).
            - Set button to "left" (default action), "right" (to open context menus), or "middle".
            - Fallback: Only pass 'x' and 'y' directly if there is no bounding box ID available.
        """
        if x < screen_width and x >= 0 and y < screen_height and y >= 0:
            await controller.click(x=x, y=y, button=button, clicks=clicks)
            return f"Successfully clicked {button} button at ({x}, {y})."
        else:
            return f"Error: Coordinates ({x}, {y}) are out of bounds for the screen size ({screen_width}, {screen_height})."

    @tool
    async def type_tool(text: str) -> str:
        """
            Types the exact provided string using the keyboard.
            CRITICAL: You MUST use the click_tool to click inside a text input field or search bar BEFORE using this tool. If a field you clicked before is highlighted, the text you type will appear there. Also, look for the blinking indicator to see which field is active right now.now. 
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