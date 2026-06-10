import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import asyncio

class YoloParser:
    def __init__(self, model_path: str):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        print(f"YoloParser initialized on {self.device}!")

    async def annotate_image(self, image_input, output_path=None, box_threshold=0.01, iou_threshold=0.1):
        """
        Runs YOLO inference on an image, draws bounding boxes with IDs, 
        and returns the annotated PIL Image.
        
        Args:
            image_input: File path (str) or PIL Image.
            output_path: Optional path to save the resulting image.
            box_threshold: Confidence threshold for bounding boxes.
            iou_threshold: Intersection over Union threshold for NMS.
            
        Returns:
            annotated_image: PIL Image with drawn bounding boxes and IDs.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path (str) or PIL Image.")

        w, h = image.size
        
        results = self.model.predict(
            source=image,
            conf=box_threshold,
            iou=iou_threshold,
            verbose=False # Keep console output clean
        )
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        box_overlay_ratio = max(w, h) / 3200
        thickness = max(int(3 * box_overlay_ratio), 2)
        text_padding = max(int(3 * box_overlay_ratio), 1)
        font_size = max(int(30 * box_overlay_ratio), 12)
        
        try:
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except IOError:
            font = ImageFont.load_default()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            text = str(i)
            
            # Draw the box border (Green)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=thickness)
            
            # Calculate text background size
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position the ID tag above the box (or inside if at the very top edge)
            label_bg = [
                x1, 
                y1 - text_height - (text_padding * 2), 
                x1 + text_width + (text_padding * 2), 
                y1
            ]
            
            # Handle edge case where box is at the absolute top of the image
            if label_bg[1] < 0:
                label_bg = [x1, y1, x1 + text_width + (text_padding * 2), y1 + text_height + (text_padding * 2)]
                draw.rectangle(label_bg, fill="green")
                draw.text((x1 + text_padding, y1 + text_padding), text, fill="white", font=font)
            else:
                draw.rectangle(label_bg, fill="green")
                draw.text((x1 + text_padding, label_bg[1] + text_padding), text, fill="white", font=font)

        
        # 6. Save the image if a path was provided
        if output_path:
            annotated_image.save(output_path)
            print(f"Annotated image saved to: {output_path}")

        return annotated_image
    



if __name__ == "__main__":
    async def test_yolo():
        parser = YoloParser(model_path="data/weights/yolo/model.pt")
        import time
        start_time = time.time()
        annotated_img = await parser.annotate_image(
            image_input="data/test_image/google_news.png", 
            output_path="data/test_image/google_news_annotated.png"
        )
        elapsed = time.time() - start_time
        print(elapsed)
                
    asyncio.run(test_yolo())
