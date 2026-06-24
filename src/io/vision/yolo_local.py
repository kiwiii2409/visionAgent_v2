import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import asyncio
import base64
import io
import os 
from typing import Dict, Optional, Tuple
import shutil

class AsyncYoloParser:
    def __init__(self, model_path: str, log_dir: Optional[str] = "data/log_images"):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)


        self.log_dir = log_dir
        self.image_counter = 0
        
        # Setup and wipe the logging directory on restart
        if self.log_dir:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[YoloClient] Log directory reset at: {self.log_dir}")
            
        print(f"YoloParser initialized on {self.device}!")

    async def query(self, image_input, output_path=None, box_threshold=0.01, iou_threshold=0.1):
        """
        Runs YOLO inference on an image, draws bounding boxes with IDs, 
        and returns the annotated PIL Image.
        
        Args:
            image_input: PIL Image or B64 string
            output_path: Optional path to save the resulting image.
            box_threshold: Confidence threshold for bounding boxes.
            iou_threshold: Intersection over Union threshold for NMS.
            
        Returns:
            annotated_image: PIL Image with drawn bounding boxes and IDs.
        """
        if isinstance(image_input, str):
            image_bytes = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path, base64 string, or PIL Image.")

        w, h = image.size
        
        results = self.model.predict(
            source=image,
            conf=box_threshold,
            iou=iou_threshold,
            verbose=False 
        )
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        box_overlay_ratio = max(w, h) / 3200
        thickness = max(int(4 * box_overlay_ratio), 3)       # Thicker box borders
        text_padding = max(int(4 * box_overlay_ratio), 4)    # More breathing room inside the label
        font_size = max(int(30 * box_overlay_ratio), 28)
        
        try:
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except IOError:
            font = ImageFont.load_default()

        button_coordinates = {}

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            button_id = str(i)
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            button_coordinates[button_id] = [center_x, center_y]

            draw.rectangle([x1, y1, x2, y2], outline="green", width=thickness)
            
            text_bbox = draw.textbbox((0, 0), button_id, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position the ID tag inside the top-right corner of the box
            label_left_edge = x2 - text_width - (text_padding * 2)
            
            # Ensure the label doesn't push past the left edge of a very narrow bounding box
            label_left_edge = max(label_left_edge, x1)
            
            label_bg = [
                label_left_edge,                           # Left edge
                y1,                                        # Top edge
                x2,                                        # Right edge
                y1 + text_height + (text_padding * 2)      # Bottom edge
            ]
            
            # Draw the background rectangle and the text
            draw.rectangle(label_bg, fill="green")
            draw.text(
                (label_left_edge + text_padding, y1 + text_padding), 
                button_id, 
                fill="white", 
                font=font
            )



        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG", quality=100) 
        b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        if self.log_dir and b64_image:
                try:
                    clean_b64 = b64_image.split(",")[-1] if "," in b64_image else b64_image
                    image_data = base64.b64decode(clean_b64)
                    
                    file_path = os.path.join(self.log_dir, f"{self.image_counter}.png")
                    with open(file_path, "wb") as f:
                        f.write(image_data)
                    
                    self.image_counter += 1
                except Exception as e:
                    print(f"[YoloClient] Failed to save image locally: {e}")
        return b64_image, button_coordinates

    



if __name__ == "__main__":
    async def test_yolo():
        parser = AsyncYoloParser(model_path="data/weights/yolo/model.pt")
        import time
        start_time = time.time()
        b64_image, button_coords = await parser.annotate_image(
            image_input="data/test_image/google_news.png", 
            output_path="data/test_image/google_news_annotated.png"
        )
        elapsed = time.time() - start_time
        print(elapsed)
                
    asyncio.run(test_yolo())
