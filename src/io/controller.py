"""
src/io/controller.py

Role:
    Mimic human control using mouse movement, clicking, and keyboard typing.

"""

import asyncio
from typing import Literal


class IOController:

    def __init__(self):
        # lazy import to ensure virtual display is set up first
        global pyautogui
        import pyautogui  
        pyautogui.FAILSAFE = True


    async def move_mouse(self, x: int, y: int, duration: float = 3.5) -> None:
        """Move the mouse cursor to absolute coordinates, top Left being (0,0)"""
        await asyncio.to_thread(pyautogui.moveTo, x, y, duration=duration)

    async def click(self, x: int | None = None, y: int | None = None, button: Literal["left", "right", "middle"] = "left", clicks: int = 1) -> None:
        """Click at specified coordinates or current position."""
        await asyncio.to_thread(
            pyautogui.click,
            x=x,
            y=y,
            clicks=clicks,
            button=button
        )

    async def scroll(self, amount: int, x: int | None = None, y: int | None = None) -> None:
        """Scroll up (positive) or down (negative) at given coordinates"""
        if x is not None and y is not None:
            await self.mouse_move(x, y, duration=0.1)

        await asyncio.to_thread(pyautogui.scroll, amount)

    async def write(self, text: str, interval: float = 0.01) -> None:
        """Type text with a delay between keystrokes"""
        await asyncio.to_thread(pyautogui.write, text, interval=interval)

    async def key_press(self, key: str) -> None:
        """Press a single key"""
        await asyncio.to_thread(pyautogui.press, key)


if __name__ == "__main__":
    async def test_controller():
        controller = IOController()
        import time
        time.sleep(5)

        await controller.mouse_move(200, 1640, duration=2.0)

        await controller.click(clicks=1)

        await controller.write("hello", interval=1)

        await controller.key_press("enter")

        await controller.scroll(-5)

    asyncio.run(test_controller())
