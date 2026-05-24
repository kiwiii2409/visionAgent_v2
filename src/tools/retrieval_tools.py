"""
src/tools/retrieval_tools.py

Role:
    provide tools for retrieving relevant information from indexed files and past task memories
"""

from langchain_chroma import Chroma
from langchain_core.tools import tool


def get_retrieval_tools(vector_store: Chroma):
    file_retriever = vector_store.as_retriever(
        search_kwargs={"k": 5, "filter": {"type": "file"}}
    )

    memory_retriever = vector_store.as_retriever(
        search_kwargs={"k": 5, "filter": {"type": "memory"}}
    )

    @tool
    async def local_file_search(query: str) -> str:
        """search pre-indexed files for relevant info"""
        results = await file_retriever.aget_relevant_documents(query)

        return "\n\n".join(f"{doc.page_content} (source: {doc.metadata.get('source', 'unknown')})" for doc in results)

    @tool
    async def task_memory_search(query: str) -> str:
        """search past task executions for similarities -> allow agent to use past successful executions as reference for current task"""
        results = await memory_retriever.aget_relevant_documents(query)

        return "\n\n".join(f"{doc.page_content} (source: {doc.metadata.get('source', 'unknown')})" for doc in results)

    return [local_file_search, task_memory_search]