"""
src/agents/vision_graph.py

Role:
    Adaptive vision-based agent loop: capture screen → think & act → execute.
    A single VLM call per iteration decides progress and next actions jointly.
"""

import io
import asyncio
import base64
from typing import Literal

from PIL import Image
from langgraph.graph import StateGraph, START, END

from src.agents.template.schema import VisionState, VisionActionSchema
from src.agents.template.prompts import get_vision_think_prompt


class VisionGraphBuilder:
    def __init__(self, vlm, mcp_tools, screen_capture, preprocessor=None, skill_manager=None, max_iterations: int = 30):
        self.vlm_thinker = get_vision_think_prompt() | vlm.with_structured_output(
            VisionActionSchema,
            method="function_calling"
        )
        self.preprocessor = preprocessor if preprocessor else None
        self.skill_manager = skill_manager
        self.mcp_tools_dict = {tool.name: tool for tool in mcp_tools}
        self.tools_info = "\n".join([f"- {t.name}: {t.description}" for t in mcp_tools])

        self.capture = screen_capture
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Node: capture_screen
    # ------------------------------------------------------------------
    async def capture_screen(self, state: VisionState) -> dict:
        """Take a screenshot, run YOLO preprocessing, encode as base64 JPEG."""
        await asyncio.sleep(3)

        img: Image.Image = await asyncio.to_thread(self.capture.full_screen)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        if self.preprocessor:
            await asyncio.sleep(10) 

            b64_annotated, coordinate_dict = await self.preprocessor.query(b64)
            if b64_annotated is not None:
                return {"screenshot_b64": b64_annotated, "coordinate_dict": coordinate_dict}
            # YOLO failed — fall through to raw screenshot

        return {"screenshot_b64": b64}

    # ------------------------------------------------------------------
    # Node: think_and_act
    # ------------------------------------------------------------------
    async def think_and_act(self, state: VisionState) -> dict:
        """Single VLM call: evaluate progress + plan next actions."""
        history_summary = self._summarize_history(state.get("action_history", []))

        # Detect repeated failures — same action+result 3 times in a row
        if self._is_stuck(state.get("action_history", [])):
            return {
                "done": True,
                "error": "Repeated same action with same result — agent is stuck.",
            }

        try:
            scratchpad = state.get("scratchpad") or "(empty)"

            # Retrieve relevant skills
            skills_text = ""
            if self.skill_manager:
                skills_text = await self.skill_manager.retrieve(state["goal"], k=1)

            result: VisionActionSchema = await self.vlm_thinker.ainvoke({
                "goal": state["goal"],
                "scratchpad": scratchpad,
                "skills": skills_text if skills_text else "(no relevant skills found)",
                "history_summary": history_summary,
                "tools_info": self.tools_info,
                "screenshot_b64": state["screenshot_b64"],
            })

            print(f"\n[Vision] Thought: {result.thought}")
            print(f"[Vision] Done: {result.done}")
            if result.scratchpad:
                print(f"[Vision] Scratchpad: {result.scratchpad}")

            if result.done:
                return {
                    "done": True,
                    "scratchpad": result.scratchpad,
                    "action_history": [{
                        "thought": result.thought,
                        "tool_name": "done",
                        "tool_args": {},
                        "result": "Goal achieved."
                    }],
                }

            actions_list = [
                {"tool_name": a.tool_name, "tool_args": a.tool_args}
                for a in result.actions
            ]
            print(f"[Vision] Actions: {[a['tool_name'] for a in actions_list]}")

            return {
                "current_plan": {
                    "thought": result.thought,
                    "actions": actions_list,
                },
                "scratchpad": result.scratchpad,
                "done": False,
            }

        except Exception as e:
            return {"error": f"VLM call failed: {e}", "done": True}

    # ------------------------------------------------------------------
    # Node: execute_action
    # ------------------------------------------------------------------
    async def execute_action(self, state: VisionState) -> dict:
        plan = state.get("current_plan")
        if not plan or not plan.get("actions"):
            return {"iterations": state.get("iterations", 0) + 1}

        thought = plan["thought"]
        actions = plan["actions"]
        new_history_entries = []

        for act in actions:
            tool_name = act["tool_name"]
            tool_args = act["tool_args"]

            # Resolve element_id → pixel coordinates
            if "element_id" in tool_args and state.get("coordinate_dict"):
                elem_id = str(tool_args["element_id"])
                coord_dict = state["coordinate_dict"]
                if elem_id in coord_dict:
                    bbox = coord_dict[elem_id]
                    tool_args["x"] = int(bbox[0])
                    tool_args["y"] = int(bbox[1])
                    del tool_args["element_id"]
                else:
                    new_history_entries.append({
                        "thought": thought,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "result": f"[Vision] Error: ID {elem_id} not found."
                    })
                    break

            tool = self.mcp_tools_dict.get(tool_name)
            if tool:
                try:
                    print(f"[Vision] Executing: {tool_name} with {tool_args}")
                    res = await tool.ainvoke(tool_args)
                    await asyncio.sleep(0.5)
                    result_str = f"Success ({tool_name}): {res}"
                except Exception as e:
                    result_str = f"Error ({tool_name}): {e}"
                    print(f"[Vision] Execute error on {tool_name}: {e}")
            else:
                result_str = f"[Vision] Error: Tool '{tool_name}' does not exist."

            new_history_entries.append({
                "thought": thought,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": result_str,
            })

            # Abort sequence on error
            if "Error" in result_str:
                print("[Vision] Aborting remaining actions due to error.")
                break

        return {
            "action_history": new_history_entries,
            "current_plan": None,
            "iterations": state.get("iterations", 0) + 1,
        }

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    def route_after_think(self, state: VisionState) -> Literal["execute_action", "__end__"]:
        if state.get("done") or state.get("error"):
            if state.get("error"):
                print(f"[Vision] Error: {state['error']}, stopping")
            else:
                print("[Vision] Goal achieved, stopping.")
            return END
        return "execute_action"

    def should_continue(self, state: VisionState) -> Literal["capture_screen", "__end__"]:
        if state.get("error"):
            print(f"[Vision] Error: {state['error']}, stopping")
            return END
        if state.get("done"):
            return END
        if state.get("iterations", 0) >= state.get("max_iterations", self.max_iterations):
            print(f"[Vision] Max iterations ({self.max_iterations}) reached, stopping")
            return END
        return "capture_screen"

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self):
        graph = StateGraph(VisionState)

        graph.add_node("capture_screen", self.capture_screen)
        graph.add_node("think_and_act", self.think_and_act)
        graph.add_node("execute_action", self.execute_action)

        graph.add_edge(START, "capture_screen")
        graph.add_edge("capture_screen", "think_and_act")

        graph.add_conditional_edges(
            "think_and_act",
            self.route_after_think,
            {"execute_action": "execute_action", END: END},
        )

        graph.add_conditional_edges(
            "execute_action",
            self.should_continue,
            {"capture_screen": "capture_screen", END: END},
        )

        return graph.compile()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _summarize_history(self, action_history: list, n: int = 10) -> str:
        if not action_history:
            return "(no actions taken yet)"

        recent = action_history[-n:]
        lines = []
        for i, a in enumerate(recent):
            tool = a.get("tool_name", "unknown")
            args = a.get("tool_args", {})
            res = a.get("result", "")
            # Truncate long results
            if len(res) > 120:
                res = res[:120] + "..."
            lines.append(f"  {i+1}. {tool}({args}) -> {res}")
        return "\n".join(lines)

    def _is_stuck(self, action_history: list, n: int = 3) -> bool:
        """Check if the last N non-done actions are identical in tool_name and result."""
        real_actions = [a for a in action_history if a.get("tool_name") != "done"]
        if len(real_actions) < n:
            return False

        last_n = real_actions[-n:]
        first = (last_n[0].get("tool_name"), last_n[0].get("result", ""))
        for a in last_n[1:]:
            if (a.get("tool_name"), a.get("result", "")) != first:
                return False
        print(f"[Vision] Detected stuck loop: {first[0]} repeated {n}x with same result.")
        return True
