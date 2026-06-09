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

from langchain_core.prompts import ChatPromptTemplate

def get_vision_planning_prompt() -> ChatPromptTemplate:
    """Requires 'goal', 'history_summary', and 'step_result'. Returns prompt for VLM action planning."""
    return ChatPromptTemplate.from_template(
        "You are an expert autonomous AI agent controlling a Linux desktop GUI (Resolution: 1920x1080). "
        "Your ultimate objective is: {goal}\n\n"
        
        "=== CURRENT STATE ===\n"
        "Recent Action History:\n{history_summary}\n\n"
        "Result of Last Action:\n{step_result}\n\n"
        
        "=== INSTRUCTIONS ===\n"
        "Analyze the provided screenshot and determine the exact next logical step. Follow these rules strictly:\n\n"
        
        "1. VISUAL CONFIRMATION: Always verify the current screen state. Look for loading indicators, error modals, or unexpected pop-ups before deciding your next move.\n"
        "2. ERROR RECOVERY: If the 'Result of Last Action' indicates a failure, or if the screen does not match your expected outcome, your NEXT action must be to recover (e.g., close an error, try an alternative search, or wait longer). Avoid jumping repeatedly between the same decisions (e.g. moving mouse multiple times in a row)\n"
        "3. PACING & LATENCY: GUI operations take real time. If an application is launching, a page is loading, or the UI transitions are not complete, you MUST invoke your 'wait' tool to allow the system to catch up.\n"
        "4. SINGLE FOCUS: Execute the next immediate logical step. Do not attempt to guess or bundle too many interactions into a single turn unless the tool explicitly supports it.\n"
        "5. TASK COMPLETION: If the overarching goal is FULLY achieved and visually confirmed on screen, DO NOT invoke any further tools. Reply ONLY with a concise text summary explaining how the task was successfully completed.\n"
        "6. Input Continuity: After clicking a text field, assume it remains active for the next step even if there is no visual focus indicator. Avoid reselecting fields that were already filled.\n"
        "7. Confident Interaction: Prefer direct actions based on the current UI state (e.g., launch apps from taskbar icons when available). Trust previous interactions, avoid unnecessary corrections or repeated actions.\n"

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