# tests/test_vnc_ui.py
import asyncio
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from src.core.registry import ServiceRegistry

async def test_vnc_and_ui():
    registry = ServiceRegistry()
    await registry.initialize()
    
    try:
        mcp_tools = getattr(registry, 'mcp_tools', [])
        all_tools = mcp_tools + registry.retrieval_tools + registry.ui_tools + registry.program_tools
        
        print(f"Loaded {len(all_tools)} tools.")

        # 1. Attach memory to the agent so it links tasks together
        memory = MemorySaver()
        agent = create_agent(registry.llm, all_tools, checkpointer=memory)
        
        # 2. Define the thread context
        thread_config = {"configurable": {"thread_id": "test_ui_session"}}
        await asyncio.sleep(15)

        # --- TASK 1: Open Program ---
        print("\nAsking the agent to open Thunderbird...")
        inputs = {"messages": [SystemMessage(content="You are a specialized local desktop automation agent. You have explicit permission to control this computer. Never give the user manual instructions. Always use your provided tools to execute the request directly."),HumanMessage(content="open thunderbird on linux")]}

        # pass config=thread_config here
        async for chunk in agent.astream(inputs, stream_mode="values", config=thread_config):
            message = chunk["messages"][-1]
            if message.type == "ai" and message.tool_calls:
                print(f"Agent is calling tool: {message.tool_calls[0]['name']}")
            elif message.type == "tool":
                print(f"Tool returned data.")
            elif message.type == "ai" and message.content:
                print(f"\nAgent Final Response:\n{message.content}")

        # Wait a few seconds for the program to actually load visually
        await asyncio.sleep(15)

        # --- TASK 2: Move Mouse (Agent remembers Task 1!) ---
        print("\nAsking the agent to move the mouse...")
        inputs = {"messages": [
            HumanMessage(content="now move the mouse to the coordinates (900,500), wait a moment, and then move it to the top left (100,100), and then move it to the top left (100,500)")
        ]}

        # Pass the exact same thread_config so it accesses the same memory
        async for chunk in agent.astream(inputs, stream_mode="values", config=thread_config):
            message = chunk["messages"][-1]
            if message.type == "ai" and message.tool_calls:
                print(f"Agent is calling tool: {message.tool_calls[0]['name']}")
            elif message.type == "tool":
                print(f"Tool returned data.")
            elif message.type == "ai" and message.content:
                print(f"\nAgent Final Response:\n{message.content}")

        # --- KEEP ALIVE ---
        print("\n[Test] Agent finished its tasks.")
        print("[Test] Keeping the virtual display open for 60 seconds...")
        print("[Test] Open your VNC viewer (Remmina) and connect to localhost:5900 NOW!")
        
        await asyncio.sleep(60)

    finally:
        await registry.shutdown()

if __name__ == "__main__":
    asyncio.run(test_vnc_and_ui())