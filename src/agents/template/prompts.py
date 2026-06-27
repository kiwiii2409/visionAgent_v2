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
        "If the local context is completely useless or the user's query is clearly about current events/external facts, set 'needs_websearch' to true to abandon local files and search the internet."
    )

def get_web_selection_prompt():
    return ChatPromptTemplate.from_template(
        """You are an expert researcher. 
        You are trying to answer the following query: "{query}"

        Here are the snippets returned from a web search:
        {search_results}

        Your task is to review these search results and select up to 2 of the most promising URLs that are most likely to contain the detailed answer to the query.
        Return ONLY the exact URLs."""
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


def get_vision_think_prompt() -> ChatPromptTemplate:
    """Unified step-by-step prompt: evaluate progress + plan next action in one call."""
    return ChatPromptTemplate.from_messages(
        [
            ("system",
            "You are an expert autonomous AI agent controlling a desktop GUI.\n"
            "Your ultimate goal is: {goal}\n\n"

            "=== AVAILABLE TOOLS ===\n"
            "{tools_info}\n\n"

            "=== APPLICATION SKILLS ===\n"
            "{skills}\n\n"

            "=== HOW TO OPERATE ===\n"
            "Each turn you will receive a fresh screenshot, recent action history, and your scratchpad.\n"
            "1. Look at the screenshot and assess progress toward the goal.\n"
            "2. If the goal is FULLY achieved (explicit visual proof on screen): set done=true, actions=[]\n"
            "3. If the goal is NOT yet achieved: set done=false, output 1-4 concrete actions to move closer.\n"
            "4. If you see important facts (prices, names, emails, URLs, file paths): write them to scratchpad so you remember them later. Format: 'key=value'.\n"
            "5. If you are stuck (same actions failing repeatedly): explain in thought, set done=true, actions=[]\n\n"

            "=== RULES ===\n"
            "- VISUAL FIRST: Always verify the screen state before acting. Check for loading indicators, errors, pop-ups.\n"
            "- WORKING MEMORY: Use scratchpad to store facts you will need later. Read prices/names/URLs from the screenshot and record them. The scratchpad persists across steps.\n"
            "- ONE STEP AT A TIME: Focus on the immediate next logical step. Do not try to plan too far ahead.\n"
            "- REAL DATA ONLY: When typing text, copy exact values (prices, names, numbers, URLs) directly from the screenshot. Never use placeholders like [actual price] or [paste here].\n"
            "- PACING: GUI operations take time. Use wait_tool after clicks that trigger loading (2-5 seconds).\n"
            "- RECOVERY: If the last action failed, try an alternative approach. Do not repeat the same failed action.\n"
            "- INPUT CONTINUITY: After clicking a text field, it stays active. Type directly without re-clicking.\n"
            "- PREFER MOUSE: Use mouse clicks over keyboard navigation when possible.\n"

            ), ("user", [
                {
                    "type": "text",
                    "text": "=== SCRATCHPAD ===\n{scratchpad}\n\n=== RECENT ACTIONS ===\n{history_summary}\n\nDecide: is the goal achieved? If not, what's the next step?"
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