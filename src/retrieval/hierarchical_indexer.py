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
from tqdm import tqdm

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate

# Prompt template for per-file summaries (was in agents/template/prompts.py)
_HINDEX_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Summarize the core purpose or content of this file in 1 short sentence. "
    "File: {filepath}\n\nContent:\n{content}"
)


class HierarchicalIndexer:
    def __init__(self, llm, vector_store: Chroma, summary_tree_path: str,  file_hashes_path:str, chunk_size: int = 500, chunk_overlap: int = 128):
        self.llm = llm
        self.vector_store = vector_store

        # check whether folder exists & set file to store hashes and file to store summareis
        self.summary_tree_path = Path(summary_tree_path) 
        self.file_hashes_path = Path(file_hashes_path)
        self.summary_tree_path.parent.mkdir(parents=True, exist_ok=True)


        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )

        self._index_lock = asyncio.Lock() # concurrency lockto avoid user triggered index interfering with background indexing

    def _insert_into_tree(self, tree: Dict[str, Any], path_tuple: tuple, summary: str):
        """Create tree strcuture and insert file + summary at respective positiion"""
        current = tree
        for part in path_tuple[:-1]:
            if part not in current:
                current[part] = {"_type": "folder", "children": {}}
            current = current[part]["children"]
        current[path_tuple[-1]] = {"_type": "file", "summary": summary}
    
    
    async def build_index(self, folders_to_index: List[str]) -> Dict:
        """ 
        Indexes all files in the specified folders. 
        Tracks still active/ existing files as well as changes using the state_file.        
        Rebuilds summary_tree and cache from scratch each run, modifies existing chromaDB to delete old files.
        To avoid redundancy, avoid overlapping folders in the folders_to_index List
        """
        async with self._index_lock:
            print(f"\n[Indexer] Updating the index for {len(folders_to_index)} folders.")
            
            # load the file which holds the hashes for each file => check for changes
            hash_file = {} 
            if self.file_hashes_path.exists():
                with open(self.file_hashes_path, "r", encoding="utf-8") as f:
                    hash_file = json.load(f)

            tree = {}
            new_chunked_docs = []
            was_updated = False # only update tree and cache if changes were made
            active_files = set() # tracks all active files and is used to remove the ones that no longer exist
            num_modified_files = 0
            
            for root_str in folders_to_index:
                root = Path(root_str).resolve()
                if not root.exists():
                    continue

                loader = DirectoryLoader(
                    root_str,
                    glob="**/*",
                    exclude=[".git/*", "__pycache__/*", "*.png", "*.jpg", "*.xopp"],
                    loader_cls=UnstructuredFileLoader,
                    recursive=True,
                    show_progress=False,
                    use_multithreading=True
                )
                raw_docs = await asyncio.to_thread(loader.load)
                
                for doc in tqdm(raw_docs, desc=f"{root_str}"):
                    file_path = Path(doc.metadata["source"]).resolve() 
                    file_path_str = str(file_path)
                    content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()

                    active_files.add(file_path_str)

                    # file already exists & is unchanged
                    if file_path_str in hash_file and content_hash == hash_file[file_path_str]["hash"]:
                        summary_result = hash_file[file_path_str]["summary"]
                    else:
                        was_updated = True
                        num_modified_files += 1 
                        # file already exists & was changed
                        if file_path_str in hash_file:
                            print(f"[Indexer] Removing outdated embeddings for {file_path_str}")
                            await asyncio.to_thread(self.vector_store._collection.delete, where={"source": file_path_str}) # remove embeddings from ChromaDB
                        
                        # avoid long llm string on empty file
                        if len(doc.page_content) == 0:
                            summary_result = "Empty File"
                        else:
                            # create summary
                            file_summarizer = _HINDEX_SUMMARY_PROMPT | self.llm
                            summary_response = await file_summarizer.ainvoke({
                                "filepath": file_path_str, 
                                "content": doc.page_content[:1500] 
                            })
                            summary_result = summary_response.content
                    
                        # update cache
                        hash_file[file_path_str] = {"hash": content_hash, "summary": summary_result}

                        # add metadata for chroma
                        doc.metadata["file_summary"] = summary_result
                        doc.metadata["type"] = "file"
                        doc.metadata["source"] = file_path_str


                        # add modified/ new document to list s.t. it's inserted into ChromaDB later
                        new_chunked_docs.extend(self.text_splitter.split_documents([doc]))


                    # get relative postiion to root as tuple 
                    relative_path_tuple = file_path.relative_to(root).parts
                    if root_str not in tree:
                        tree[root_str] = {"_type": "root", "children": {}}
                    
                    # update tree
                    self._insert_into_tree(
                        tree[root_str]["children"], 
                        relative_path_tuple, 
                        summary_result
                    )

            if num_modified_files:
                print(f"[Indexer] Total of {num_modified_files} modified files detected & updated")

            # find paths which are no longer active (deleted or renamed)
            removed_paths = [path for path in hash_file.keys() if path not in active_files] 
            if removed_paths:
                was_updated = True
                print(f"[Indexer] Removing {len(removed_paths)} inactive files from Chroma")
                for removed_file in removed_paths:
                    try:
                        await asyncio.to_thread(self.vector_store._collection.delete, where={"source": removed_file})
                    except Exception:
                        pass  # ignore error thats thrown in case the chunks were already deleted 
                    hash_file.pop(removed_file)
            
            if was_updated:
                # save json tree
                with open(self.summary_tree_path, "w", encoding="utf-8") as f:
                    json.dump(tree, f, indent=2, ensure_ascii=False)
                print(f"[Indexer] Summary tree saved to {self.summary_tree_path}")

                with open(self.file_hashes_path, "w", encoding="utf-8") as f:
                    json.dump(hash_file, f, indent=2, ensure_ascii=False)
                print(f"[Indexer] Status file saved to {self.file_hashes_path}")
                if new_chunked_docs:
                    # save chunks to chroma
                    await self.vector_store.aadd_documents(new_chunked_docs)
                    print("[Indexer] Updated chunks saved to Chroma")
            else:
                print("[Indexer] No change detected")

