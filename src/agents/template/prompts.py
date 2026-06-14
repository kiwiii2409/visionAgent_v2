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


def get_macro_planning_prompt() -> ChatPromptTemplate:
    """Requires 'goal' as parameter. Returns prompt for high-level task breakdown."""
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert high-level task planner for a GUI automation agent.\n"
         "Your job is to break down the user's overarching goal into a sequence of small, highly specific, and actionable subgoals.\n"
         "Each subgoal should represent a distinct phase of the task (e.g., '1. Open Web Browser', '2. Navigate to google.com', '3. Search for the query').\n"
         "Keep the subgoals concise and achievable in a few steps."
        ),
        ("user", "Goal: {goal}")
    ])


def get_vision_evaluation_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert, objective visual QA evaluator for an autonomous computer agent. "
         "Your sole job is to look at a screenshot and determine if a specific subgoal has been successfully accomplished.\n\n"
         "CRITICAL RULES:\n"
         "1. Be strict and objective. Do not assume the action succeeded unless there is explicit visual proof. Provide the evidence\n"
         "2. Look for concrete visual cues: changed button states, opened menus, specific text appearing on screen, URL changes, or success modals.\n"
         "3. If a subgoal requires finding or opening a setting, the setting's specific menu or window MUST be open and visible. Seeing a search result or launcher icon is NOT sufficient.\n"
         "4. If the screenshot looks like it is still loading, or you cannot definitively tell if the goal is met, mark it as NOT achieved.\n"
        ),
        ("human", [
            {
                "type": "text", 
                "text": "Please evaluate the screen state.\n\nSubgoal to evaluate: {current_subgoal}"
            },
            {
                "type": "image_url", 
                "image_url": {"url": "data:image/jpeg;base64,{screenshot_b64}"}
            }
        ])
    ])

def get_vision_planning_prompt() -> ChatPromptTemplate:
    """Requires 'goal', 'history_summary', 'tools_info', and 'screenshot_b64'. Returns prompt for VLM action planning."""
    return ChatPromptTemplate.from_messages(
        [
            ("system",
            "You are an expert autonomous AI agent controlling a desktop GUI.\n"
            "Your ultimate objective is: {goal}\n\n"
            "Your CURRENT SUBGOAL is: {current_subgoal}\n"
            "Focus entirely on completing this CURRENT SUBGOAL. Do not worry about the later steps yet.\n\n"
            
            "=== AVAILABLE TOOLS ===\n"
            "{tools_info}\n\n"

            "=== INSTRUCTIONS ===\n"
            "Analyze the provided screenshot and determine the exact next logical step. Follow these rules strictly:\n\n"
            
            "1. VISUAL CONFIRMATION: Always verify the current screen state. Look for loading indicators, error modals, or unexpected pop-ups before deciding your next move.\n"
            "2. ERROR RECOVERY: If the Recent Action History indicates a failure, or if the screen does not match your expected outcome, your NEXT action must be to recover (e.g., close an error, try an alternative search, or wait longer). Avoid jumping repeatedly between the same decisions (e.g., moving mouse multiple times in a row).\n"
            "3. PACING & LATENCY: GUI operations take real time. If an application is launching, a page is loading, or the UI transitions are not complete, you MUST invoke your 'wait' tool to allow the system to catch up.\n"
            "4. SINGLE FOCUS: Execute the next immediate logical step. Do not attempt to guess or bundle too many interactions into a single turn if you expect the UI to change during the interactions.\n"
            "5. INPUT CONTINUITY: After clicking a text field, assume it remains active for the next step even if there is no visual focus indicator. Avoid reselecting fields that were already filled and avoid navigating using keys, prefer the mouse!\n"
            "6. OS AWARENESS: Find out which OS you are on using visual cues. Open the respective system menu and use the search function to open apps or switch windows.\n"

            ), ("user", [
                {
                    "type": "text", 
                    "text": "=== CURRENT STATE ===\nRecent Action History:\n{history_summary}\n\nOutput your reasoning and the next tool(s) to execute.\n "
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": "data:image/jpeg;base64,{screenshot_b64}"}
                }
            ])
        ]
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