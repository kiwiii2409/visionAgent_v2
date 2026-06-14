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
    sources: List[str]
    iterations: int                # Safety bound — stops explore loop after N rounds
    max_iterations: int            # Configurable limit, set by graph builder
    

class EvaluationSchema(BaseModel):
    is_sufficient: bool = Field(description="True if the context fully answers the query and the surrounding files do not provide additional information, False otherwise.")
    reasoning: str = Field(description="Very short and concise explanation of why. Max 1 sentence.")

class FileSelectionSchema(BaseModel):
    selected_files: List[str] = Field(description="List of absolute file paths to read next. Max 3.")

class TaskRoutingSchema(BaseModel):
    task_type: Literal["question", "task"] = Field(description="question if it's a pure knowledge-retrieval query, task if it involves interacting or manipulating the system")
    reasoning: str = Field(description="Very short and concise explanation of why. Max 1 sentence.")


class FinalAnswerSchema(BaseModel):
    answer: str = Field(description="The synthesized final answer to the user's query.")
    sources: List[str] = Field(description="List of absolute file paths that were used as sources for this answer.")


# --- Vision Agent Schemas ---

class ToolCallSchema(BaseModel):
    tool_name: str = Field(description="The exact name of the tool to execute. Use 'done' if the goal is fully achieved.")
    tool_args: dict = Field(default_factory=dict, description="Parameters for the tool. Use 'element_id' instead of exact x,y-pixelvalues for UI interactions.  Empty if tool_name is 'done'.")

class VisionActionSchema(BaseModel):
    """Structured output from VLM: a sequence of actions to take on the desktop."""
    thought: str = Field(description="Brief reasoning about what you see and why these actions are the logical next steps. Max 3 sentences.")
    actions: List[ToolCallSchema] = Field(description="List of tools (max. 4) to execute in sequence. Only chain tools which don't trigger reactions in the UI")


class VisionState(TypedDict):
    """State for the vision-based agent loop."""
    goal: str
    screenshot_b64: str | None
    coordinate_dict: Dict | None
    action_history: Annotated[list, operator.add]  # List[dict], append-only trajectory
    current_plan: Dict | None
    done: bool
    iterations: int
    max_iterations: int
    error: str | None