"""
src/io/vlm/omniparser.py

Role:
    Connects to OmniParser, returns a labeled image + dictionary with {button_id: coordinates}
    Requires OmniParser to run on server + port-forwarding 
"""

from dotenv import load_dotenv
import time
import io
import os
import base64
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from gradio_client import Client, handle_file
from pathlib import Path

def convert_image_to_b64(file_path):
    """reads an image from the path and returns it's b64 value"""
    if not file_path or not os.path.exists(file_path):
        print(f"[ImgPath2B64] File not found: {file_path}")
        return None
        
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


class AsyncOmniParserClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.client = Client(self.base_url)

    async def query(self, image_path: Path, box_treshold=0.05, 	iou_threshold=0.1, use_paddleocr=True, imgsz=640, api_name="/process") -> Tuple[str,Tuple[str,str]]:
        """Preprocesses an image using OmniParser_v2; takes the image_path and returns the path for the annotated image and a dictionary of descriptions + coordinates for each labeled button"""
        image_str = str(image_path)

        if not image_str.startswith(('http://', 'https://')):
            if not os.path.exists(image_str):
                raise FileNotFoundError(f"Local image file not found at: {image_str}")
            
        try:
            raw_result = await asyncio.to_thread(
                self.client.predict,
                image_input=handle_file(image_str),
                box_threshold=box_treshold,
                iou_threshold=iou_threshold,
                use_paddleocr=use_paddleocr,
                imgsz=imgsz,
                api_name=api_name,
            )
            annotated_image_path = raw_result[0]
            parsed_data = raw_result[1]
            return annotated_image_path, parsed_data        
        except Exception as e:
             print(f"[OmniParserClient] Annotating the image failed with: {e}")


    

if __name__ == "__main__":
    async def test_omni():
        parser = AsyncOmniParserClient(base_url="http://127.0.0.1:7861")
        img_sizes = [640]#, 1280, 1920]        
        test_url = "/home/kiwiii/CodingProjects/visionAgent_v2/data/test_image/overlaying_fileexplorer.png"
        timing = {}
        for imgsz in img_sizes:
            for _ in range(1):
                timing[f"time_{imgsz}"] = []
                start_time = time.time()
                annotated_img, parsed_json = await parser.query(
                    image_path=test_url,
                    box_treshold=0.05,
                    iou_threshold=0.1,
                    use_paddleocr=False,
                    imgsz=imgsz
                )
                elapsed = time.time() - start_time
                timing[f"time_{imgsz}"].append(elapsed)
                print(f"Image saved to path: {annotated_img}")
                # print("Parsed Data:")
                print(str(parsed_json)) 

        for name, times in timing.items():
            total_time = sum(times)
            print(total_time)
            # print(f"{name:<25} | Req 1: {times[0]:>5.2f}s | Req 2: {times[1]:>5.2f}s | Req 3: {times[2]:>5.2f}s | Total: {total_time:>5.2f}s")

    asyncio.run(test_omni())