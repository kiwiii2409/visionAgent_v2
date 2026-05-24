"""
src/io/capture.py

Role:
    Screen capture utilities for the VLM pipeline.
    Takes screenshots and provides image preprocessing (resize, region selection).
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator

import mss
from PIL import Image


class ScreenCapture:
    def __init__(self) -> None:
        self.screenCapture = mss.mss()

    def full_screen(self) -> Image.Image:
        """Capture full screen"""
        monitor = self.screenCapture.monitors[1]
        sct_img = self.screenCapture.grab(monitor)
        
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

    def region(self, x: int, y: int, width: int, height: int) -> Image.Image:
        """Capture a specific screen region. (x,y) as top-left corner"""
        bbox = {"top": y, "left": x, "width": width, "height": height}
        sct_img = self.screenCapture.grab(bbox)
        
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

    async def stream(self, fps: float = 1.0) -> AsyncIterator[Image.Image]:
        """Screenshots at a regular interval, fps limited by processing speed"""
        interval = 1.0 / fps
        
        while True:
            start_time = asyncio.get_event_loop().time()
            
            # Yield the current screen frame
            yield self.full_screen()
            
            # Calculate how long to sleep to maintain the target FPS
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0.0, interval - elapsed)
            
            await asyncio.sleep(sleep_time)

    def save(self, image: Image.Image, path: Path) -> Path:
        """Save a captured image to disk"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        image.save(path)
        return path



if __name__ == "__main__":
    async def test_capture():
        capture = ScreenCapture()
        
        full_img = capture.full_screen()
        save_path = Path("./data/test_images/test_full_screen.jpg")
        capture.save(full_img, save_path)
        print(f"Caputre saved at {save_path.absolute()}")

        frame_count = 0
        async for frame in capture.stream(fps=2.0):
            print(f"Frame {frame_count}, size: {frame.size}")
            frame_count += 1
            if frame_count >= 3:
                break
                
    asyncio.run(test_capture())