"""
src/agents/templates/schema.py

Role:
Collects all Schemas and states for langchain
"""

from typing import TypedDict, List, Literal, Annotated, Set, Dict
from pydantic import BaseModel, Field
import operator

class SearchState(TypedDict):
    query: str
    context_blocks: List[str]      # Stores text from Chroma, tree maps, and full files
    known_file_paths: List[str]    # Keeps track of which files we retrieved chunks from/ read
    explored_subtrees: Set[str]    # Keeps track of which parts of the tree we explored
    final_answer: str
    sources: List[Dict[str,str]]   # enriched sources for Google-like view (instead of path, include summary, name and path)
    file_summaries: Dict[str, str] # Lookup dict for paths -> summaries
    web_summaries: Dict[str, Dict[str, str]]  # Lookup dict for url -> summaries

    iterations: int                # Safety bound — stops explore loop after N rounds
    max_iterations: int            # Configurable limit, set by graph builder
    use_websearch: bool             # allow the agent to access websites
    needs_websearch_flag: bool      # set by the evaluation_node if context is completely useless

class EvaluationSchema(BaseModel):
    reasoning: str = Field(description="Very short and concise explanation of why. Max 1 sentence.")
    is_sufficient: bool = Field(description="True if the context fully answers the query and the surrounding files do not provide additional information, False otherwise.")
    needs_websearch: bool = Field(False, description="True ONLY IF the provided context is completely useless and the answer requires external/current knowledge")

class WebSelectionSchema(BaseModel):
    selected_urls: List[str] = Field(
        default_factory=list, 
        description="A list of up to 2 highly relevant URLs extracted from the provided context results to read in full. Leave empty if no results are relevant."
    )

class FileSelectionSchema(BaseModel):
    selected_files: List[str] = Field(description="List of absolute file paths to read next. Max 3.")

class TaskRoutingSchema(BaseModel):
    task_type: Literal["question", "task"] = Field(description="question if it's a pure knowledge-retrieval query, task if it involves interacting or manipulating the system")
    reasoning: str = Field(description="Very short and concise explanation of why. Max 1 sentence.")


class FinalAnswerSchema(BaseModel):
    answer: str = Field(description="The synthesized final answer to the user's query.")
    sources: List[str] = Field(description="List all of the absolute file paths and urls that were RELEVANT for this answer. If no information was retrieved from this source, DO NOT list it")


# --- Vision Agent Schemas ---

class ToolCallSchema(BaseModel):
    tool_name: str = Field(description="The exact name of the tool to execute.")
    tool_args: dict = Field(default_factory=dict, description="Parameters for the tool. Use 'element_id' instead of exact (x,y)-pixel-values for UI interactions.  Empty if tool_name is 'done'.")

class VisionActionSchema(BaseModel):
    """Structured output from VLM: whether the goal is done, reasoning, and next actions."""
    thought: str = Field(description="Brief reasoning about current screen state, what has been done, and the next logical step. Max 3 sentences.")
    done: bool = Field(description="True if the goal is fully achieved based on visual evidence on screen. False if more actions are needed.")
    actions: List[ToolCallSchema] = Field(default_factory=list, description="Next tools to execute. Empty if done=True. Max 4.")
    scratchpad: str | None = Field(default=None, description="Important facts to remember across steps (prices, names, URLs, emails). Write key=value pairs like 'BTC=53195.36 EUR'. This persists and will be shown to you next iteration.")




class VisionState(TypedDict):
    goal: str
    screenshot_b64: str | None
    coordinate_dict: Dict | None
    action_history: Annotated[list, operator.add]
    scratchpad: str | None      # Cross-iteration working memory (facts, prices, names)
    current_plan: Dict | None
    done: bool
    iterations: int
    max_iterations: int
    error: str | None
    use_websearch: bool             # allow the searchAgent (TODO make tool) to use the web

