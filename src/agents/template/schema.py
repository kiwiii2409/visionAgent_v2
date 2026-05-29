"""
src/agents/templates/schema.py

Role:
Collects all Schemas and states for langchain
"""

from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field

class SearchState(TypedDict):
    query: str
    context_blocks: List[str]      # Stores text from Chroma, tree maps, and full files
    known_file_paths: List[str]    # Keeps track of what we already found/read
    final_answer: str
    sources: List[str]

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