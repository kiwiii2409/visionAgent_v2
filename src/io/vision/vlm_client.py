"""
src/io/vlm/client.py

Role:
    Interface to the local/ api VLM model, handles image analysis
    Currently unused
"""

from dotenv import load_dotenv
import time
import io
import os
import base64
import asyncio
from typing import List, Dict, Any, Optional
from PIL import Image
from openai import AsyncOpenAI


class VLMClient:
    def __init__(self, model_name: str, base_url: str, api_key: str) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.client = AsyncOpenAI(
            api_key=api_key if api_key else "dummy-key",
            base_url=self.base_url
        )

    async def query(self, image: Image.Image, prompt: str, response_format: Optional[Dict[str, Any]] = None, temperature: float = 0.3) -> str:
        """Send an image and text prompt to the VLM and return the generated text"""
        b64_img = self._encode_image(image)
        payload = self._format_payload(b64_img, prompt)

        kwargs = {
            "model": self.model_name,
            "messages": payload,
            "temperature": temperature
        }

        if response_format:
            kwargs["response_format"] = response_format
        try:
            completion = await asyncio.wait_for(
                self.client.chat.completions.create(**kwargs),
                timeout=5.0 # TODO is this too low?
            )
            return completion.choices[0].message.content
        except asyncio.TimeoutError:
            print("[VLMClient] API call timed out after 5 seconds.")
            return ""
        except Exception as e:
            print(f"[VLMClient] Failed to analyze image with error : {e}")
            return '{"thought": "API call failed.", "actions": []}'

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64-encoded string"""
        buffered = io.BytesIO()

        # convert to RGB to avoid issues with alpha
        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(buffered, format="JPEG", quality=95)

        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def _format_payload(self, image_b64: str, prompt: str) -> dict:
        """Build the JSON payload expected by the local model endpoint."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
        return messages


load_dotenv()


if __name__ == "__main__":
    async def test_vlm():
        models_to_test = [
            # "gpt-5.4-nano-2026-03-17",
            "gpt-5.4-mini",
            # "gpt-5-nano-2025-08-07",
            # "gpt-4.1-nano-2025-04-14"
        ]

        timings = {}

        TEST_PROMPT = """
            Return which button number to click for the following task: {task_description}
        """

        try:
            test_image_desktop = Image.open("data/test_image/google_news_annotated.png")

        except Exception as e:
            print(f"Warning: Could not load cat image. ({e})")
            return

        for model_name in models_to_test:
            print("\n" + "="*50)
            print(f"Testing Model: {model_name}")
            print("=" * 50)

            client = VLMClient(
                model_name=model_name,
                base_url="https://api.openai.com/v1",
                api_key=os.environ.get("OPENAI_API_KEY")
            )

            timings[model_name] = []

            for _ in range(1):
                print("Sending Image")
                try:
                    user_task = "Which button number should i click to close the window"
                    start_time = time.time()
                    response = await client.query(
                        image=test_image_desktop,
                        prompt=TEST_PROMPT.replace(
                            "{task_description}", user_task)
                    )
                    elapsed = time.time() - start_time
                    timings[model_name].append(elapsed)

                    print(f"Response ({elapsed:.2f}s)")
                    print(response)
                except Exception as e:
                    print(f"Failed: {e}")

        # # Final Comparison Print
        # print("\n\n" + "="*50)
        # print("Results:")

        # for model, times in timings.items():
        #     total_time = sum(times)
        #     print(
        #         f"{model:<25} | Req 1: {times[0]:>5.2f}s | Req 2: {times[1]:>5.2f}s | Req 3: {times[2]:>5.2f}s | Total: {total_time:>5.2f}s")
# Results:
# gpt-5.4-nano-2026-03-17   | Req 1:  2.15s | Req 2:  1.55s | Req 3:  1.85s | Total:  5.56s
# gpt-5.4-mini-2026-03-17   | Req 1:  1.56s | Req 2:  2.61s | Req 3:  1.74s | Total:  5.90s
# gpt-5-nano-2025-08-07     | Req 1: 25.13s | Req 2: 38.42s | Req 3: 24.21s | Total: 87.76s
# gpt-4.1-nano-2025-04-14   | Req 1:  2.12s | Req 2:  1.59s | Req 3:  2.11s | Total:  5.82s

    asyncio.run(test_vlm())
