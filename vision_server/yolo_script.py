import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import base64
import io

class YoloParser:
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        print(f"[YoloParser] Initialized on {self.device}!")

    async def annotate_image(self, image_input: str, box_threshold=0.01, iou_threshold=0.1):
        if "," in image_input and image_input.startswith("data:image"):
            image_input = image_input.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")

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

        return b64_image, button_coordinates
