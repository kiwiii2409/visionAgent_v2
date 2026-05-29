"""
src/agents/task_router.py

Role:
Queries LLM to decide which agent to call -> pure search vs. vision-based 
"""

from src.agents.template.schema import TaskRoutingSchema
from src.agents.template.prompts import get_task_routing_prompt

async def route_query(query: str, llm) -> str:
    """
    Classify the user query into search vs. task
    """
    router_chain = get_task_routing_prompt() | llm.with_structured_output(TaskRoutingSchema)
    
    result = await router_chain.ainvoke({"query": query})
    
    print(f"\n[Router] Decision: {(result.task_type).upper()}")
    
    return result.task_type