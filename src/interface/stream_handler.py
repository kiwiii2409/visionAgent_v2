import json
import asyncio
from src.core.registry import ServiceRegistry

async def generate_chat_stream(prompt: str, registry: ServiceRegistry):
    """Routes the query and yields Server-Sent Events (JSON lines) for the frontend."""
    from src.agents.task_router import route_query
    
    task_type = await route_query(prompt, registry.llm)
    yield json.dumps({"type": "init", "mode": task_type}) + "\n"
    
    try:
        async with asyncio.timeout(300):
            if task_type == "question":
                async for chunk in _stream_search_agent(prompt, registry):
                    yield chunk
            else:
                async for chunk in _stream_vision_agent(prompt, registry):
                    yield chunk
    except asyncio.TimeoutError:
        yield json.dumps({"type": "error", "content": "Agent timed out after 5 minutes."}) + "\n"
    except Exception as e:
        yield json.dumps({"type": "error", "content": f"Agent Error: {str(e)}"}) + "\n"

async def _stream_search_agent(prompt: str, registry: ServiceRegistry):
    initial_state = {
        "query": prompt,
        "context_blocks": [],
        "known_file_paths": [],
        "final_answer": "",
        "sources": [],
        "iterations": 0,
        "max_iterations": registry.settings.max_search_iterations
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

async def _stream_vision_agent(prompt: str, registry: ServiceRegistry):
    vision_state = {
        "goal": prompt,
        "screenshot_b64": None,
        "action_history": [],
        "step_result": "",
        "done": False,
        "iterations": 0,
        "max_iterations": registry.settings.max_iterations,
        "error": None,
    }

    async for event in registry.vision_agent.astream(vision_state):
        for node_name, state_update in event.items():
            if not state_update: continue

            if node_name == "capture_screen":
                yield json.dumps({"type": "tool", "name": "Capturing screen"}) + "\n"
                await asyncio.sleep(0.3)
                yield json.dumps({"type": "tool_done"}) + "\n"

            elif node_name == "plan_action":
                history = state_update.get("action_history", [])
                if history:
                    last = history[-1]
                    if last.get("type") == "tool_call":
                        tool_names = [call.get("name", "unknown") for call in last.get("calls", [])]
                        label = f"Plan: Using tools ({', '.join(tool_names)})"
                    elif last.get("type") == "done":
                        label = f"Plan: Done — {last.get('reasoning', '')[:60]}"
                    else:
                        label = "Plan: Processing..."                                
                    
                    yield json.dumps({"type": "tool", "name": label}) + "\n"
                    await asyncio.sleep(0.3)
                    yield json.dumps({"type": "tool_done"}) + "\n"

            elif node_name == "execute_action":
                step = state_update.get("step_result", "")
                if state_update.get("done", False):
                    yield json.dumps({"type": "msg", "content": f"Task complete after {state_update.get('iterations', 0)} steps."}) + "\n"
                elif step:
                    yield json.dumps({"type": "tool", "name": step}) + "\n"
                    await asyncio.sleep(0.2)
                    yield json.dumps({"type": "tool_done"}) + "\n"