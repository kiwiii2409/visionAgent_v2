import os
import time
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import asyncio
import base64
import io
import shutil


class AsyncYoloParser:
    def __init__(self, model_path: str, log_dir: str | None = "data/log_images"):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[AsyncYoloParser] Weights not found: {model_path}")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.failure_count = 0
        self.total_calls = 0
        self.image_counter = 0

        self.log_dir = log_dir
        if self.log_dir:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[AsyncYoloParser] Log directory reset at: {self.log_dir}")

        print(f"[AsyncYoloParser] Initialized | device={self.device} | path={model_path}")

    async def query(self, image_input: str, box_threshold: float = 0.01, iou_threshold: float = 0.1):
        """
        Compatible interface with AsyncYoloClient.query().
        Takes a base64 image string, returns (annotated_b64, coordinate_dict).
        Returns (None, None) on failure so the caller can fall back.
        Runs inference in a thread pool to avoid blocking the event loop.
        """
        try:
            return await asyncio.to_thread(
                self._annotate_sync,
                image_input=image_input,
                box_threshold=box_threshold,
                iou_threshold=iou_threshold,
            )
        except Exception as e:
            print(f"[AsyncYoloParser] query failed: {e}")
            return None, None

    async def annotate_image(self, image_input, output_path=None, box_threshold=0.01, iou_threshold=0.1):
        """Async wrapper. Runs the heavy sync work in a thread pool."""
        return await asyncio.to_thread(
            self._annotate_sync,
            image_input=image_input,
            output_path=output_path,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
        )

    def _annotate_sync(self, image_input, output_path=None, box_threshold=0.01, iou_threshold=0.1):
        """
        Synchronous core: decode, run YOLO inference, draw bounding boxes, encode.
        Runs in a thread pool so the event loop stays free.
        """
        if isinstance(image_input, str):
            image_bytes = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a base64 string or PIL Image.")

        t0 = time.time()
        try:
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
            thickness = max(int(4 * box_overlay_ratio), 3)
            text_padding = max(int(4 * box_overlay_ratio), 4)
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
                label_left_edge = max(label_left_edge, x1)

                label_bg = [
                    label_left_edge,
                    y1,
                    x2,
                    y1 + text_height + (text_padding * 2)
                ]

                draw.rectangle(label_bg, fill="green")
                draw.text(
                    (label_left_edge + text_padding, y1 + text_padding),
                    button_id,
                    fill="white",
                    font=font
                )

            if output_path:
                annotated_image.save(output_path)
                print(f"Annotated image saved to: {output_path}")

            buffered = io.BytesIO()
            annotated_image.save(buffered, format="JPEG", quality=85)
            b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            elapsed = time.time() - t0
            self.total_calls += 1
            self.failure_count = 0
            if elapsed > 2.0:
                print(f"[AsyncYoloParser] WARNING: slow inference ({elapsed:.1f}s)")

            if self.log_dir:
                try:
                    file_path = os.path.join(self.log_dir, f"{self.image_counter}.png")
                    annotated_image.save(file_path)
                    self.image_counter += 1
                except Exception as e:
                    print(f"[AsyncYoloParser] Failed to save log image: {e}")

            return b64_image, button_coordinates

        except Exception as e:
            self.failure_count += 1
            self.total_calls += 1
            elapsed = time.time() - t0
            print(f"[AsyncYoloParser] ERROR (#{self.failure_count}, {elapsed:.1f}s): {e}")
            raise



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
