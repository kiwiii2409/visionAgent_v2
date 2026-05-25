"""
src/agents/templates/prompts.py
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
    """ requires 'query', 'known_files' and 'dir_contents' as parameter"""
    return ChatPromptTemplate.from_template(
        "User Query: {query}\n"
        "Files we already know about: {known_files}\n\n"
        "Here are the contents of neighboring directories:\n{dir_contents}\n\n"
        "Based on the query, select up to 3 files from these directories that are MOST likely to contain the missing answers. "
        "Do not select files we already know about."
    )

def get_synthesis_prompt() -> ChatPromptTemplate:
    """ requires 'query' and 'context' as parameter"""

    return ChatPromptTemplate.from_template(
        "You are a local filesystem assistant. Answer the query using ONLY the provided local file context. "
        "Always cite your sources using the file paths.\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}"
    )