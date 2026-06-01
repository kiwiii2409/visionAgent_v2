"""
src/agents/templates/schema.py

Role:
Collects all Schemas and states for langchain
"""

from typing import TypedDict, List, Literal, Annotated
from pydantic import BaseModel, Field
import operator

class SearchState(TypedDict):
    query: str
    context_blocks: List[str]      # Stores text from Chroma, tree maps, and full files
    known_file_paths: List[str]    # Keeps track of what we already found/read
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

class VisionActionSchema(BaseModel):
    """Structured output from VLM: a single action to take on the desktop."""
    thought: str = Field(description="Brief reasoning about what you see and why this action is the right next step. Max 2 sentences.")
    done: bool = Field(description="True only if the user's goal has been fully achieved. False if more actions are needed.")
    action_type: Literal["move", "click", "type", "key", "launch", "wait", "done"] = Field(
        description="The type of action to execute: move, click, type, key, launch, wait, or done."
    )
    params: dict = Field(
        default_factory=dict,
        description="Parameters for the action. move: {x, y}; click: {button}; type: {text}; key: {key}; launch: {command}; wait: {seconds}; done: {}"
    )


class VisionState(TypedDict):
    """State for the vision-based agent loop."""
    goal: str
    screenshot_b64: str | None
    action_history: Annotated[list, operator.add]  # List[dict], append-only trajectory
    step_result: str
    done: bool
    iterations: int
    max_iterations: int
    error: str | None