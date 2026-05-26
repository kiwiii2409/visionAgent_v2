
import asyncio
import json
from src.core.registry import ServiceRegistry

async def main():
    registry = ServiceRegistry()
    await registry.initialize()
    
    try:
        query = "List all the specific tools available inside the ui_tools suite, and tell me exactly what arguments the click tool takes."
        print(f"User Query: {query}\n")
        

        initial_state = {
            "query": query,
            "context_blocks": [],
            "known_file_paths": [],
            "final_answer": ""
        }
        
        
        final_state = await registry.search_agent.ainvoke(initial_state)
        

        print("=" * 50)
        print("FINAL ANSWER: ")
        print(final_state["final_answer"])

        print("=" * 50)
        print("Explored file paths: ")
        print(final_state["known_file_paths"])


        print()
        print(final_state["context_blocks"])

    finally:
        await registry.shutdown()

if __name__ == "__main__":
    asyncio.run(main())