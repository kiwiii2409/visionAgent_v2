"""
src/agents/template/prompts.py

Role: 
Collects all prompts
"""

from langchain_core.prompts import ChatPromptTemplate

def get_evaluation_prompt() -> ChatPromptTemplate:
    """ requires 'query' and 'context' as parameter"""
    return ChatPromptTemplate.from_template(
        "User Query: {query}\n\nRetrieved Context:\n{context}\n\n"
        "Does the retrieved context contain enough specific information to fully and accurately answer the user's query? "
        "Do not guess. If critical details are missing, say it is insufficient."
    )

def get_file_selection_prompt() -> ChatPromptTemplate:
    """ requires 'query', 'known_files' and 'context' as parameter"""
    return ChatPromptTemplate.from_template(
        "User Query: {query}\n"
        "Files with currently retrieved snippets: {known_files}\n\n"
        "Context (including partial code chunks and local directory maps):\n{context}\n\n"
        "Based on the query and the provided maps, select up to 3 files that are MOST likely to contain the missing answers. "
        "NOTE: The context only contains small snippets of the 'known_files'. "
        "If you suspect the missing information is located elsewhere inside one of those exact same files, "
        "you SHOULD select it here so we can read the entire file. You may also select completely new files from the maps. "
        "Return absolute paths."
    )

def get_synthesis_prompt() -> ChatPromptTemplate:
    """ requires 'query' and 'context' as parameter"""
    return ChatPromptTemplate.from_template(
        "You are a local filesystem assistant. Answer the query using ONLY the provided local file context.\n"
        "Keep your answer short and concise, without skipping relevant information.\n"
        "Use rich Markdown formatting (code blocks with language tags, inline code for variables/paths, bold text, and lists) to maximize readability.\n"
        "Extract any file paths you used to answer the query into the separate sources list.\n"
        "Do not include them in the main response unless explicitly asked\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}"
    )

def get_vision_planning_prompt() -> ChatPromptTemplate:
    """Requires 'goal', 'history_summary', and 'step_result'. Returns prompt for VLM action planning."""
    return ChatPromptTemplate.from_template(
        "You are controlling a Linux desktop (1920x1080). Your goal is: {goal}\n\n"
        "Recent actions taken:\n{history_summary}\n\n"
        "Last step result: {step_result}\n\n"
        "Look at the screenshot and decide the NEXT logical step. Always consider the time each operation takes and add a wait-toolcall if necesssary"
        "INSTRUCTIONS:\n"
        "- If you need to interact with the screen or system, invoke the appropriate provided tool(s).\n"
        "- If the goal is FULLY achieved and no further action is needed, DO NOT call any tools. "
        "Simply reply with a brief text summary explaining how the goal was accomplished."
    )


def get_task_routing_prompt() -> ChatPromptTemplate:
    """ requires 'query' as parameter"""
    return ChatPromptTemplate.from_template(
        "You are a highly efficient routing agent for a local desktop automation system.\n"
        "Your job is to classify the user's query into one of two categories:\n\n"
        "1. 'question': The user wants to find a file, read code, understand architecture, or search for text. Any task that only requires reading files falls under this category. (Requires knowledge retrieval from the filesystem).\n"
        "2. 'task': The user wants the agent to take physical action, such as clicking, typing, looking at the screen, or opening an application. (Requires system manipulation).\n\n"
        "Query: {query}"
    )