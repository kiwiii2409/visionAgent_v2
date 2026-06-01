"""
src/retrieval/hierarchical_indexer.py

Role: 
Indexes documents + creates a tree including summaries for each file => used by agent to select additional files
"""

import os
import json
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate

# Prompt template for per-file summaries (was in agents/template/prompts.py)
_HINDEX_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Summarize the core purpose of this file in 1 short sentence. "
    "File: {filepath}\n\nContent:\n{content}"
)


class HierarchicalIndexer:
    def __init__(self, llm, vector_store: Chroma, summary_tree_path: str,  chunk_size: int = 500, chunk_overlap: int = 128):
        self.llm = llm
        self.vector_store = vector_store

        # summary_tree_path is the full path to the tree JSON file
        self.summary_tree_path = Path(summary_tree_path)
        self.summary_tree_path.parent.mkdir(parents=True, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )

    def _insert_into_tree(self, tree: Dict[str, Any], path_tuple: tuple, summary: str):
        """Create tree strcuture and insert file + summary at respective positiion"""
        current = tree
        for part in path_tuple[:-1]:
            if part not in current:
                current[part] = {"_type": "folder", "children": {}}
            current = current[part]["children"]
        current[path_tuple[-1]] = {"_type": "file", "summary": summary}
    
    
    async def build_index(self, folders_to_index: List[str]):
        print(f"\n[Indexer] Starting Indexing + File-Tree Summary Generation")
        
        tree = {}
        all_chunked_docs = []

        for root_str in folders_to_index:
            root = Path(root_str)
            if not root.exists():
                continue

            loader = DirectoryLoader(
                root_str,
                glob="**/*",
                exclude=[".git/*", "__pycache__/*", "*.png", "*.jpg"],
                loader_cls=UnstructuredFileLoader,
                recursive=True,
                show_progress=True,
                use_multithreading=True
            )
            raw_docs = await asyncio.to_thread(loader.load)
                
            for doc in raw_docs:
                filepath = Path(doc.metadata["source"])
                
                # avoid long llm string on empty file
                if len(doc.page_content) == 0:
                    summary_result = "Empty File"
                else:
                    # create summary
                    file_summarizer = _HINDEX_SUMMARY_PROMPT | self.llm
                    summary_response = await file_summarizer.ainvoke({
                        "filepath": str(filepath), 
                        "content": doc.page_content[:1500] 
                    })
                    summary_result = summary_response.content
                
                # get relative postiion to root as tuple 
                relative_path_tuple = filepath.relative_to(root).parts
                if root_str not in tree:
                    tree[root_str] = {"_type": "root", "children": {}}
                
                self._insert_into_tree(
                    tree[root_str]["children"], 
                    relative_path_tuple, 
                    summary_result
                )

                # add summary + type as metadata for Chroma retrieval filters
                doc.metadata["file_summary"] = summary_result
                doc.metadata["type"] = "file"

            # gather all chunks from all roots
            all_chunked_docs.extend(self.text_splitter.split_documents(raw_docs))

        # save json tree
        with open(self.summary_tree_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        print(f"[Indexer] JSON tree saved to {self.summary_tree_path}")

        # save chunks to chroma
        await self.vector_store.aadd_documents(all_chunked_docs)
        print("[Indexer] Chunks saved to Chroma")


