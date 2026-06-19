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
from typing import Literal, Optional
import tempfile

from PIL import Image
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from src.agents.template.schema import VisionState, VisionActionSchema, MacroPlanSchema, VisionEvaluationSchema
from src.agents.template.prompts import get_vision_planning_prompt, get_macro_planning_prompt, get_vision_evaluation_prompt


class VisionGraphBuilder:
    def __init__(self, vlm, mcp_tools, screen_capture, preprocessor = None, max_iterations: int = 10):
        self.vlm_macro_planner = get_macro_planning_prompt() | vlm.with_structured_output(
            MacroPlanSchema, 
            method="function_calling"
        )

        self.vlm_evaluator = get_vision_evaluation_prompt() | vlm.with_structured_output(
            VisionEvaluationSchema,
            method="function_calling"
        )
        self.vlm_action_planner = get_vision_planning_prompt() | vlm.with_structured_output(
            VisionActionSchema, 
            method="function_calling"
        )        
        self.preprocessor = preprocessor if preprocessor else None
        self.mcp_tools_dict = {tool.name : tool for tool in mcp_tools}
        self.tools_info = "\n".join([f"- {t.name}: {t.description}" for t in mcp_tools])

        self.capture = screen_capture
        self.max_iterations = max_iterations


    # ------------------------------------------------------------------
    # Node: macro_plan 
    # ------------------------------------------------------------------
    async def macro_plan(self, state: VisionState) -> dict:
        """Break down the main goal into a series of actionable subgoals."""
        try:
            print(f"[Vision] Generating macro plan for goal: {state['goal']}")
            plan: MacroPlanSchema = await self.vlm_macro_planner.ainvoke({"goal": state["goal"]})
            print(f"[Vision] Generated Subgoals:")
            for i, sg in enumerate(plan.subgoals):
                print(f"  {i+1}. {sg}")
                
            return {
                "subgoals": plan.subgoals,
                "current_subgoal_index": 0,
                "subgoal_done": False
            }
        except Exception as e:
            return {"error": f"Macro planner failed: {e}", "done": True}
        

    # ------------------------------------------------------------------
    # Node: capture_screen
    # ------------------------------------------------------------------
    async def capture_screen(self, state: VisionState) -> dict:
        """Take a screenshot and encode as base64 JPEG for the VLM."""
        # await asyncio.sleep(3) 

        # if state.get("iterations", 0) == 0:
        #     print("[Vision] Initialization: Opening OS Start Menu...")
            
        #     keyboard_tool = self.mcp_tools_dict.get("key_press_tool") 
        #     if keyboard_tool:
        #         try:
        #             await keyboard_tool.ainvoke({"key": "win"}) 
        #             await asyncio.sleep(0.8) 
        #         except Exception as e:
        #             print(f"[Vision] Warning: Could not open OS menu: {e}")

        await asyncio.sleep(3) 

        img: Image.Image = await asyncio.to_thread(self.capture.full_screen)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=100)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        if self.preprocessor:
            b64_annotated, coordinate_dict = await self.preprocessor.query(b64)
            return {"screenshot_b64": b64_annotated, "coordinate_dict": coordinate_dict}
        
        return {"screenshot_b64": b64, "coordinate_dict": {}}


    async def evaluate_state(self, state: VisionState) -> dict:
        """Evaluate if the current subgoal is complete based on the latest screenshot."""
        
        # if state.get("iterations", 0) == 0:
        #     return {"subgoal_done": False}

        subgoals = state.get("subgoals", [state["goal"]])
        idx = state.get("current_subgoal_index", 0)
        current_subgoal = subgoals[idx] if idx < len(subgoals) else state["goal"]

        print(f"\n[Vision] --- Working on Subgoal {idx+1}/{len(subgoals)}: {current_subgoal} ---")
        print(f"[Vision] Evaluating state for Subgoal: {current_subgoal}")
        
        try:
            eval_result: VisionEvaluationSchema = await self.vlm_evaluator.ainvoke({
                "current_subgoal": current_subgoal,
                "screenshot_b64": state["screenshot_b64"]
            })
            
            print(f"[Vision] Evaluation Result: {eval_result.is_subgoal_achieved}. Reasoning: {eval_result.reasoning}")
            
            if eval_result.is_subgoal_achieved:
                return {
                    "subgoal_done": True,
                    "action_history": [{"thought": eval_result.reasoning, "tool_name": "done", "tool_args": {}, "result": f"Subgoal verified complete: {current_subgoal}"}]
                }
            return {"subgoal_done": False}
            
        except Exception as e:
            print(f"[Vision] Evaluator failed: {e}")
            return {"subgoal_done": False}
   
    # ------------------------------------------------------------------
    # Node: plan_action
    # ------------------------------------------------------------------
    async def plan_action(self, state: VisionState) -> dict:
        """Send (goal + screenshot + history) to the VLM, parse the action JSON."""
        history_summary = self._summarize_history(state.get("action_history", []))

        subgoals = state.get("subgoals", [state["goal"]])
        idx = state.get("current_subgoal_index", 0)
        current_subgoal = subgoals[idx] if idx < len(subgoals) else state["goal"]

        try:
            action_plan: VisionActionSchema = await self.vlm_action_planner.ainvoke({
                    "goal": state["goal"],
                    "current_subgoal": current_subgoal, 
                    "history_summary": history_summary,
                    "tools_info": self.tools_info,
                    "screenshot_b64": state["screenshot_b64"],
                    "coordinate_dict": state["coordinate_dict"]
                })
            print("\n\n History Summary")
            print(history_summary)
            print("\n\n")
            
        except Exception as e:
            return {"error": f"VLM call failed: {e}", "done": True}


        actions_list = [{"tool_name": a.tool_name, "tool_args": a.tool_args} for a in action_plan.actions]
        tool_names = [a.tool_name for a in action_plan.actions]
        print(f"[Vision] Plan: Sequenced Tools: {tool_names}.")

        return {
            "current_plan": {
                "thought": action_plan.thought,
                "actions": actions_list
            },
            "done": False
        }


    # ------------------------------------------------------------------
    # Node: execute_action
    # ------------------------------------------------------------------
    async def execute_action(self, state: VisionState) -> dict:
        plan = state.get("current_plan")
        if not plan or not plan.get("actions"):
            return {"iterations": state.get("iterations", 0) + 1}

        thought = plan["thought"]
        actions = plan["actions"]
        print(f"[Vision] Thought: {thought}")
        print(f"[Vision] Action: {actions}")
        new_history_entries = []
        
        for act in actions:
            tool_name = act["tool_name"]
            tool_args = act["tool_args"]

            if "element_id" in tool_args and state.get("coordinate_dict"):
                elem_id = str(tool_args["element_id"])
                coord_dict = state["coordinate_dict"]
                if elem_id in coord_dict:
                    bbox = coord_dict[elem_id]
                    tool_args["x"] = int(bbox[0])
                    tool_args["y"] = int(bbox[1])

                    del tool_args["element_id"] 
                else:
                    new_history_entries.append({"result": f"[Vision] Error: ID {elem_id} not found."})
                    break
                

            tool = self.mcp_tools_dict.get(tool_name)
            if tool:
                try:
                    print(f"[Vision] Executing Tool: {tool_name} with args {tool_args}")
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
                "result": result_str
            })
            
            if "Error" in result_str:
                print(f"[Vision] Aborting remainder of sequence due to error.")
                break

        return {
            "action_history": new_history_entries,
            "current_plan": None,
            "iterations": state.get("iterations", 0) + 1,
        }

    async def next_subgoal(self, state: VisionState) -> dict:
        """Transitions to the next subgoal in the macro plan."""
        idx = state.get("current_subgoal_index", 0) + 1
        subgoals = state.get("subgoals", [])
        
        if idx < len(subgoals):
            print(f"[Vision] Moving to next subgoal: {subgoals[idx]}")
            return {
                "current_subgoal_index": idx,
                "subgoal_done": False
            }
        
        return {"done": True}
   
    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------
    def should_continue(self, state: VisionState) -> Literal["capture_screen", "next_subgoal", "__end__"]:
        if state.get("error"):
            print(f"[Vision] Error: {state['error']}, stopping")
            return END
        
        if state.get("done"): 
            return END

        if state.get("iterations", 0) >= state.get("max_iterations", self.max_iterations):
            print(f"[Vision] Max iterations reached, stopping")
            return END

        return "capture_screen"

    def route_after_evaluation(self, state: VisionState) -> Literal["plan_action", "next_subgoal", "__end__"]:
        if state.get("subgoal_done"):
            idx = state.get("current_subgoal_index", 0)
            subgoals = state.get("subgoals", [])
            if idx >= len(subgoals) - 1:
                print(f"[Vision] All subgoals verified as completed!")
                return END
            else:
                return "next_subgoal"
        return "plan_action"
    

    # ------------------------------------------------------------------
    # Build the graph
    # ------------------------------------------------------------------
    def build(self):
        graph = StateGraph(VisionState)

        graph.add_node("macro_plan", self.macro_plan)         
        graph.add_node("capture_screen", self.capture_screen)
        graph.add_node("evaluate_state", self.evaluate_state) 
        graph.add_node("plan_action", self.plan_action)
        graph.add_node("execute_action", self.execute_action)
        graph.add_node("next_subgoal", self.next_subgoal)    

        graph.add_edge(START, "macro_plan")
        graph.add_edge("macro_plan", "capture_screen")
        
        graph.add_edge("capture_screen", "evaluate_state")
        
        graph.add_conditional_edges(
            "evaluate_state",
            self.route_after_evaluation,
            {
                "plan_action": "plan_action", 
                "next_subgoal": "next_subgoal", 
                END: END
            },
        )
        
        graph.add_edge("plan_action", "execute_action")

        graph.add_conditional_edges(
            "execute_action",
            self.should_continue, 
            {
                "capture_screen": "capture_screen", 
                END: END
            },
        )
        
        graph.add_edge("next_subgoal", "capture_screen")

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
            lines.append(f"  {i+1}. Action: {tool} with {args} -> Result: {res}")
        return "\n".join(lines)
