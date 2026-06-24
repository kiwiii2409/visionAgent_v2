import json
import asyncio
from src.core.registry import ServiceRegistry

# async def generate_chat_stream(prompt: str, registry: ServiceRegistry):
#     """Routes the query and yields Server-Sent Events (JSON lines) for the frontend."""
#     from src.agents.task_router import route_query
    
#     task_type = await route_query(prompt, registry.llm)
#     yield json.dumps({"type": "init", "mode": task_type}) + "\n"
    
#     try:
#         async with asyncio.timeout(300):
#             if task_type == "question":
#                 async for chunk in _stream_search_agent(prompt, registry):
#                     yield chunk
#             else:
#                 async for chunk in _stream_vision_agent(prompt, registry):
#                     yield chunk
#     except asyncio.TimeoutError:
#         yield json.dumps({"type": "error", "content": "Agent timed out after 5 minutes."}) + "\n"
#     except Exception as e:
#         yield json.dumps({"type": "error", "content": f"Agent Error: {str(e)}"}) + "\n"

async def _stream_search_agent(prompt: str, use_websearch:bool, registry: ServiceRegistry):
    initial_state = {
        "query": prompt,
        "context_blocks": [],
        "known_file_paths": [],
        "explored_subtrees": set(),      
        "final_answer": "",
        "sources": [],
        "file_summaries": {},            
        "iterations": 0,
        "max_iterations": registry.settings.max_search_iterations,
        "use_websearch": use_websearch
    }
    
    async for event in registry.search_agent.astream(initial_state):
        for node_name, state_update in event.items():
            if not state_update: continue

            if node_name == "initial_retrieval":
                yield json.dumps({"type": "tool", "name": "Retrieving file chunks and maps"}) + "\n"
                await asyncio.sleep(0.5)
                yield json.dumps({"type": "tool_done"}) + "\n"

            elif node_name == "evaluate_context":
                is_sufficient = state_update.get("is_sufficient_flag")
                status = "Context sufficient." if is_sufficient else "Missing info. Expanding search..."
                yield json.dumps({"type": "tool", "name": status}) + "\n"
                await asyncio.sleep(0.5)
                yield json.dumps({"type": "tool_done"}) + "\n"

            elif node_name == "explore_additional_files":
                files = state_update.get("known_file_paths", [])
                if files:
                    yield json.dumps({"type": "tool", "name": f"Reading file: {files[-1]}"}) + "\n"
                    await asyncio.sleep(0.5)
                    yield json.dumps({"type": "tool_done"}) + "\n"

            elif node_name == "synthesize_answer":
                if answer := state_update.get("final_answer", ""):
                    yield json.dumps({"type": "msg", "content": answer, "sources": state_update.get("sources", [])}) + "\n"

async def _stream_vision_agent(prompt: str, use_websearch:bool, registry: ServiceRegistry):
    vision_state = {
        "goal": prompt,
        "screenshot_b64": None,
        "coordinate_dict": None,
        "action_history": [],
        "current_plan": None,
        "done": False,
        "iterations": 0,
        "max_iterations": registry.settings.max_iterations,
        "error": None,
        "use_websearch": use_websearch
    }

    async for event in registry.vision_agent.astream(vision_state):
        for node_name, state_update in event.items():
            if not state_update: continue

            if node_name == "capture_screen":
                yield json.dumps({"type": "tool", "name": "Capturing screen"}) + "\n"
                await asyncio.sleep(0.3)
                yield json.dumps({"type": "tool_done"}) + "\n"
            elif node_name == "plan_action":
                plan = state_update.get("current_plan")
                if plan:
                    actions = plan.get('actions', [])
                    tools_str = ", ".join([a.get('tool_name', 'unknown') for a in actions])
                    thought = plan.get('thought', '')[:60]
                    label = f"Plan: {tools_str} ({thought}...)"
                    yield json.dumps({"type": "tool", "name": label}) + "\n"
                    await asyncio.sleep(0.3)
                    yield json.dumps({"type": "tool_done"}) + "\n"
                elif state_update.get("done"):
                    history = state_update.get("action_history", [])
                    if history:
                        last = history[-1]
                        label = f"Plan: Done — {last.get('thought', '')[:60]}"
                        yield json.dumps({"type": "tool", "name": label}) + "\n"
                        await asyncio.sleep(0.3)
                        yield json.dumps({"type": "tool_done"}) + "\n"
                        yield json.dumps({"type": "msg", "content": "Task complete."}) + "\n"

            elif node_name == "execute_action":
                # LangGraph yields the *delta* for Annotated fields, so history here is a list of the *new* actions.
                history_delta = state_update.get("action_history", [])
                for step in history_delta:
                    result_msg = step.get("result", "")
                    if result_msg:
                        yield json.dumps({"type": "tool", "name": result_msg}) + "\n"
                        await asyncio.sleep(0.2)
                        yield json.dumps({"type": "tool_done"}) + "\n"