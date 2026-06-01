"""
src/agents/vision_graph.py

Role:
    Vision-based agent loop: capture screen → plan next action → execute → verify.
    Flow: capture_screen → plan_action → execute_action → (loop until done or max).

Uses a VLM to "see" the screen and decide mouse/keyboard actions.
"""

import io
import json
import re
import asyncio
import base64
import subprocess
from typing import Literal

from PIL import Image
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from src.agents.template.schema import VisionState
from src.agents.template.prompts import get_vision_planning_prompt


class VisionGraphBuilder:
    def __init__(self, vlm, screen_capture, io_controller, max_iterations: int = 10):
        self.vlm = vlm
        self.capture = screen_capture
        self.controller = io_controller
        self.max_iterations = max_iterations
        self.planning_prompt = get_vision_planning_prompt()

    # ------------------------------------------------------------------
    # Node: capture_screen
    # ------------------------------------------------------------------
    async def capture_screen(self, state: VisionState) -> dict:
        """Take a screenshot and encode as base64 JPEG for the VLM."""
        img: Image.Image = await asyncio.to_thread(self.capture.full_screen)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"screenshot_b64": b64}

    # ------------------------------------------------------------------
    # Node: plan_action
    # ------------------------------------------------------------------
    async def plan_action(self, state: VisionState) -> dict:
        """Send (goal + screenshot + history) to the VLM, parse the action JSON."""
        history_summary = self._summarize_history(state.get("action_history", []))
        step_result = state.get("step_result", "(no previous step)")

        prompt_text = self.planning_prompt.format(
            goal=state["goal"],
            history_summary=history_summary,
            step_result=step_result,
        )

        # Build multimodal message: text prompt + base64 screenshot
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{state['screenshot_b64']}"
                },
            },
        ])

        try:
            response = await self.vlm.ainvoke([msg])
            raw_text = response.content if hasattr(response, "content") else str(response)
            print(f"[Vision] VLM raw response ({len(raw_text)} chars): {raw_text[:200]}")
        except Exception as e:
            print(f"[Vision] VLM call failed: {e}")
            return {
                "action_history": [{"action_type": "done", "params": {}, "reasoning": f"VLM call failed: {e}"}],
                "done": True,
                "error": f"VLM call failed: {e}",
                "step_result": f"VLM error: {e}",
            }

        parsed = self._parse_action_json(raw_text)
        action_type = parsed.get("action_type", "done")
        params = parsed.get("params", {})
        reasoning = parsed.get("thought", "")

        print(f"[Vision] Plan: {action_type} | {reasoning[:80]}")

        record = {
            "action_type": action_type,
            "params": params,
            "reasoning": reasoning,
        }

        return {
            "action_history": [record],
            "done": bool(parsed.get("done", False)),
            "step_result": reasoning,
        }

    # ------------------------------------------------------------------
    # Node: execute_action
    # ------------------------------------------------------------------
    async def execute_action(self, state: VisionState) -> dict:
        """Execute the planned action via IOController."""
        history = state.get("action_history", [])
        if not history:
            return {"step_result": "No action to execute", "iterations": state.get("iterations", 0) + 1}

        last_action = history[-1]
        action_type = last_action["action_type"]
        params = last_action.get("params", {})

        try:
            if action_type == "move":
                x, y = int(params.get("x", 0)), int(params.get("y", 0))
                await self.controller.move_mouse(x, y, duration=0.8)
                result = f"Moved mouse to ({x}, {y})"

            elif action_type == "click":
                button = params.get("button", "left")
                await self.controller.click(button=button)
                result = f"Clicked {button} button"

            elif action_type == "type":
                text = params.get("text", "")
                await self.controller.write(text)
                result = f"Typed: {text[:50]}"

            elif action_type == "key":
                key = params.get("key", "")
                if "+" in key:
                    import pyautogui
                    parts = [k.strip() for k in key.split("+")]
                    await asyncio.to_thread(pyautogui.hotkey, *parts)
                else:
                    await self.controller.key_press(key)
                result = f"Pressed key: {key}"

            elif action_type == "launch":
                cmd = params.get("command", "")
                subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                result = f"Launched: {cmd}"

            elif action_type == "wait":
                seconds = float(params.get("seconds", 1.0))
                await asyncio.sleep(min(seconds, 5.0))
                result = f"Waited {min(seconds, 5.0):.1f}s"

            elif action_type == "done":
                result = "Goal achieved — done"

            else:
                result = f"Unknown action: {action_type}"

        except Exception as e:
            result = f"Action failed: {e}"
            print(f"[Vision] Execute error: {e}")

        print(f"[Vision] Execute: {result}")
        new_iterations = state.get("iterations", 0) + 1

        return {
            "step_result": result,
            "iterations": new_iterations,
        }

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------
    def should_continue(self, state: VisionState) -> Literal["capture_screen", "__end__"]:
        if state.get("done"):
            return END
        if state.get("iterations", 0) >= state.get("max_iterations", self.max_iterations):
            print(f"[Vision] Max iterations reached, stopping")
            return END
        if state.get("error"):
            print(f"[Vision] Error: {state['error']}, stopping")
            return END
        return "capture_screen"

    # ------------------------------------------------------------------
    # Build the graph
    # ------------------------------------------------------------------
    def build(self):
        graph = StateGraph(VisionState)

        graph.add_node("capture_screen", self.capture_screen)
        graph.add_node("plan_action", self.plan_action)
        graph.add_node("execute_action", self.execute_action)

        graph.add_edge(START, "capture_screen")
        graph.add_edge("capture_screen", "plan_action")
        graph.add_edge("plan_action", "execute_action")

        graph.add_conditional_edges(
            "execute_action",
            self.should_continue,
            {"capture_screen": "capture_screen", END: END},
        )

        return graph.compile()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _summarize_history(self, action_history: list, n: int = 5) -> str:
        """Take the last N actions and format them for the prompt."""
        if not action_history:
            return "(no actions taken yet)"
        recent = action_history[-n:]
        lines = [
            f"  {i+1}. [{a['action_type']}] {a.get('reasoning', '')}"
            for i, a in enumerate(recent)
        ]
        return "\n".join(lines)

    def _parse_action_json(self, raw_text: str) -> dict:
        """Robust JSON extraction from VLM output (may contain markdown fences)."""
        # Try direct parse first
        try:
            return json.loads(raw_text.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from code fence
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try extracting first { ... } block
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Give up — return safe fallback
        print(f"[Vision] Failed to parse VLM output: {raw_text[:200]}")
        return {"thought": "JSON parse failed", "done": True, "action_type": "done", "params": {}}
