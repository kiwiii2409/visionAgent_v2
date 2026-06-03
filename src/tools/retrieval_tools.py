"""
src/tools/retrieval_tools.py

Role:
    provide tools for retrieving relevant information from indexed files and past task memories
"""

from langchain_chroma import Chroma
from langchain_core.tools import tool


def get_retrieval_tools(vector_store: Chroma):
    """
        Used to retrieve past successful executions logs to use as few-shot prompting or call searchAgent to retrieve information (not implemented yet)
    """
    # No type filter on file_retriever: the hierarchical indexer currently
    # stores all docs without a "type" metadata field. Filtering on {"type":"file"}
    # would silently return zero results. When task_memory writes "type":"memory"
    # documents, we can re-enable the filter.
    file_retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    memory_retriever = vector_store.as_retriever(
        search_kwargs={"k": 4, "filter": {"type": "memory"}}
    )

    @tool
    async def local_file_search(query: str) -> str:
        """Search pre-indexed files for relevant info"""
        results = await file_retriever.aget_relevant_documents(query)

        return "\n\n".join(f"{doc.page_content} (source: {doc.metadata.get('source', 'unknown')})" for doc in results)

    @tool
    async def task_memory_search(query: str) -> str:
        """Search past task executions for similarities -> allow vision agent to use past successful executions as reference for current task"""
        results = await memory_retriever.aget_relevant_documents(query)

        return "\n\n".join(f"{doc.page_content} (source: {doc.metadata.get('source', 'unknown')})" for doc in results)

    return [local_file_search, task_memory_search]