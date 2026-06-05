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
    def __init__(self, vlm, mcp_tools, screen_capture, max_iterations: int = 10):
        self.vlm = vlm.bind_tools(mcp_tools)

        self.mcp_tools_dict = {tool.name : tool for tool in mcp_tools}
        
        self.capture = screen_capture
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
        except Exception as e:
            return {"error": f"VLM call failed: {e}", "done": True}

        # Case: VLM wants to call tools
        if response.tool_calls:
            print(f"[Vision] Plan: Triggering {len(response.tool_calls)} tools.")
            return {
                "action_history": [{"type": "tool_call", "calls": response.tool_calls}],
                "done": False,
            }
        else:
            # no tools => likely done; TODO: maybe need to add waiting tool s.t. llm can set wait times instead of repeatedly calling tools while the program is loading
            print(f"[Vision] Plan: Done. Reasoning: {response.content[:100]}")
            return {
                "action_history": [{"type": "done", "reasoning": response.content}],
                "done": True,
                "step_result": response.content
            }


    # ------------------------------------------------------------------
    # Node: execute_action
    # ------------------------------------------------------------------
    async def execute_action(self, state: VisionState) -> dict:
        """Execute the planned action via IOController."""
        history = state.get("action_history", [])
        if not history or history[-1]["type"] == "done":
            return {"iterations": state.get("iterations", 0) + 1}

        tool_calls = history[-1]["calls"]
        results = []

        for tc in tool_calls:
            
            tool = self.mcp_tools_dict.get(tc["name"])
            if tool:
                try:
                    print(f"[Vision] Executing Tool: {tc['name']} with args {tc['args']}")
                    res = await tool.ainvoke(tc["args"]) 
                    results.append(f"Success ({tc['name']}): {res}")
                except Exception as e:
                    results.append(f"Error ({tc['name']}): {e}")
                    print(f"[Vision] Execute error on {tc['name']}: {e}")
            else:
                results.append(f"Error: Tool '{tc['name']}' does not exist.")

        return {
            "step_result": "\n".join(results),
            "iterations": state.get("iterations", 0) + 1,
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
        lines = []
        for i, a in enumerate(recent):
            if a.get("type") == "tool_call":
                names = [tool_call["name"] for tool_call in a["calls"]]
                lines.append(f"  {i+1}. Used tools: {', '.join(names)}")
            else:
                lines.append(f"  {i+1}. Finished: {a.get('reasoning', '')}")
        return "\n".join(lines)

