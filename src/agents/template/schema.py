"""
src/agents/templates/schema.py
"""

from typing import TypedDict, List
from pydantic import BaseModel, Field

# --- Graph State ---
class SearchState(TypedDict):
    query: str
    context_blocks: List[str]      # Stores text from Chroma and read files
    known_file_paths: List[str]    # Keeps track of what we already found/read
    directories_to_explore: List[str] 
    files_to_read: List[str]
    final_answer: str

# --- LLM Structured Output Schemas ---
class EvaluationSchema(BaseModel):
    is_sufficient: bool = Field(description="True if the context fully answers the query, False otherwise.")
    reasoning: str = Field(description="Brief explanation of why.")

class FileSelectionSchema(BaseModel):
    selected_files: List[str] = Field(description="List of absolute file paths to read next. Max 3.")