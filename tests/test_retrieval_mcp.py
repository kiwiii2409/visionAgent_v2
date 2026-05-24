# tests/test_retrieval.py
import asyncio
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from src.core.registry import ServiceRegistry

async def test_retrieval_and_mcp():
    registry = ServiceRegistry()
    await registry.initialize()
    
    try:
        mcp_tools = getattr(registry, 'mcp_tools', [])
        all_tools = mcp_tools + registry.retrieval_tools + registry.ui_tools
        
        print(f"Loaded {len(all_tools)} tools.")
        for tool in all_tools:
            print(f"- {tool.name}: {tool.description}")

        agent = create_agent(registry.llm, all_tools)

        print("\nAsking the agent to search...")
        inputs = {"messages": [HumanMessage(content="Which services are registered in the registry.py")]}

        async for chunk in agent.astream(inputs, stream_mode="values"):
            message = chunk["messages"][-1]
            if message.type == "ai" and message.tool_calls:
                print(f"Agent is calling tool: {message.tool_calls[0]['name']}")
            elif message.type == "tool":
                print(f"Tool returned data.")
            elif message.type == "ai" and message.content:
                print(f"\nAgent Final Response:\n{message.content}")

    finally:
        await registry.shutdown()

if __name__ == "__main__":
    asyncio.run(test_retrieval_and_mcp())