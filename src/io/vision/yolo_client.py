import os
import time
import base64
import asyncio
import requests
import shutil
from typing import Dict, Optional, Tuple



class AsyncYoloClient:
    def __init__(self, base_url, log_dir: Optional[str] = "data/log_images") -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/annotate"

        self.log_dir = log_dir
        self.image_counter = 0
        
        # Setup and wipe the logging directory on restart
        if self.log_dir:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[YoloClient] Log directory reset at: {self.log_dir}")
        
    async def query(
        self, 
        image_input: str, 
        box_threshold: float = 0.01,  
        iou_threshold: float = 0.1
    ) -> Tuple[Optional[str], Optional[Dict[str, list]]]:
        """
        receives a b64 image and returns the annotated b64 string+ the coordinate dictionary
        """
        payload = {
            "image_b64": image_input,
            "box_threshold": box_threshold,
            "iou_threshold": iou_threshold
        }

        try:
            response = await asyncio.to_thread(
                requests.post,
                self.endpoint,
                json=payload,
                timeout=30 
            )
            
            if response.status_code != 200:
                print(f"[YoloClient] Server returned error {response.status_code}: {response.text}")
                return None, None
            
            data = response.json()
            print(f"[YoloClient] Successfully annotated image")
            annotated_b64 = data.get("annotated_b64")
            # Save the annotated image to disk
            if self.log_dir and annotated_b64:
                try:
                    # Strip data URI prefix if it exists (e.g., "data:image/png;base64,...")
                    clean_b64 = annotated_b64.split(",")[-1] if "," in annotated_b64 else annotated_b64
                    image_data = base64.b64decode(clean_b64)
                    
                    file_path = os.path.join(self.log_dir, f"{self.image_counter}.png")
                    with open(file_path, "wb") as f:
                        f.write(image_data)
                    
                    self.image_counter += 1
                except Exception as e:
                    print(f"[YoloClient] Failed to save image locally: {e}")

            return data.get("annotated_b64"), data.get("button_coordinates", {})
                    
        except Exception as e:
             print(f"[YoloClient] Annotating the image failed with connection error: {e}")
             return None, None



if __name__ == "__main__":
    async def test_yolo_client():
        parser = AsyncYoloClient(base_url="http://127.0.0.1:8020")
        test_file = "data/test_image/google_news.png"
        
        if not os.path.exists(test_file):
            print(f"Test image not found at {test_file}. Please update the path.")
            return

        timing = {"time_yolo": []}
        print(f"Testing YoloClient against {parser.base_url}...")
        
        for _ in range(3):
            start_time = time.time()
            
            annotated_b64, parsed_json = await parser.query(
                image_input=test_file,
                box_threshold=0.05,
                iou_threshold=0.1
            )
            
            elapsed = time.time() - start_time
            timing["time_yolo"].append(elapsed)
            
            if parsed_json:
                print(f"Successfully detected {len(parsed_json)} elements.")
            else:
                print("Failed to get a response from the server.")

        for name, times in timing.items():
            if not times: continue
            total_time = sum(times)
            print(f"{name:<25} | Req 1: {times[0]:>5.2f}s | Req 2: {times[1]:>5.2f}s | Req 3: {times[2]:>5.2f}s | Total: {total_time:>5.2f}s")

    asyncio.run(test_yolo_client())