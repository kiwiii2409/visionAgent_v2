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

from tqdm.asyncio import tqdm as async_tqdm 
from tqdm import tqdm 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.file_loader import AsyncFileLoader
import time

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
            
            docs_to_summarize = []
            paths_to_delete = []

            file_loader = AsyncFileLoader() # init file loader for indexing
            loaded_files = await file_loader.load(folders_to_index) # returns tuple (root, file_path)

            time_start = time.time()
            # file loading and gathering changes (wihtout applying them)

            for root_str, doc in loaded_files:
                file_path_str = doc.metadata["source"]
                file_path = Path(file_path_str).resolve() 
                root = Path(root_str).resolve()
                
                content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                active_files.add(file_path_str)

                relative_path_tuple = file_path.relative_to(root).parts
                if root_str not in tree:
                    tree[root_str] = {"_type": "root", "children": {}}

                # file already exists & is unchanged => immediate update of tree
                if file_path_str in hash_file and content_hash == hash_file[file_path_str]["hash"]:
                    summary_result = hash_file[file_path_str]["summary"]
                    # update tree
                    self._insert_into_tree(
                        tree[root_str]["children"], 
                        relative_path_tuple, 
                        summary_result
                    )
                else:
                    # mark for processing (update or add new; done later as batch to speed it up)
                    docs_to_summarize.append((doc, file_path_str, root_str, relative_path_tuple, content_hash))
                    
                    if file_path_str in hash_file: # updated files
                        paths_to_delete.append(file_path_str)

            time_elapsed = time.time() - time_start
            print(f"[Indexer] Loaded {len(active_files)} Files in {time_elapsed:.2f}s")

            # find filepaths which are no longer active (deleted or renamed)   
            removed_paths = [path for path in hash_file.keys() if path not in active_files]
            paths_to_delete.extend(removed_paths)

            # if false, we skip step of rewriting the tree at the end
            was_updated = len(docs_to_summarize) > 0 or len(paths_to_delete) > 0

            time_start = time.time()
            # remove any old embeddings (file was deleted/ updated)
            if paths_to_delete:
                print(f"[Indexer] Removing {len(removed_paths)} inactive files from Chroma")
                try:
                    # batch using $in isntead of loop
                    await asyncio.to_thread(
                        self.vector_store._collection.delete, 
                        where={"source": {"$in": paths_to_delete}}
                    )
                except Exception as e:
                    pass

                # clean up hashfile
                for path in removed_paths:
                    hash_file.pop(path, None)

            time_elapsed = time.time() - time_start
            print(f"[Indexer] Deleted chunks of {len(removed_paths)} inactive files in {time_elapsed:.2f}s")


            time_start = time.time()
            # concurrent API requests to speed up, set semaphore to num of concurrent requests allowed
            if docs_to_summarize:
                print(f"[Indexer] Total of {len(docs_to_summarize)} modified/ new files detected")
                file_summarizer = _HINDEX_SUMMARY_PROMPT | self.llm
                sem = asyncio.Semaphore(10)

                async def process_doc(task_data):
                    doc, file_path_str, _, _, _ = task_data
                    
                    async with sem:
                        if len(doc.page_content) == 0:
                            summary_result = "Empty File"
                        else:
                            summary_response = await file_summarizer.ainvoke({
                                "filepath": file_path_str, 
                                "content": doc.page_content[:1500] 
                            })
                            summary_result = summary_response.content
                    
                    return task_data, summary_result
                

                tasks = [process_doc(data) for data in docs_to_summarize]
                results = await async_tqdm.gather(*tasks, desc="Summarizing Files")

                # parse reuslts amd gather in list to add all at once
                for task_data, summary_result in results:
                    doc, file_path_str, root_str, relative_path_tuple, content_hash = task_data
                    
                    # update hash
                    hash_file[file_path_str] = {"hash": content_hash, "summary": summary_result}
                    self._insert_into_tree(tree[root_str]["children"], relative_path_tuple, summary_result)

                    # update metadata and split into chunks
                    doc.metadata["file_summary"] = summary_result
                    doc.metadata["type"] = "file"
                    doc.metadata["source"] = file_path_str
                    new_chunked_docs.extend(self.text_splitter.split_documents([doc]))
            time_elapsed = time.time() - time_start
            print(f"[Indexer] Summarized and gathered results of {len(docs_to_summarize)} files in {time_elapsed:.2f}s")

            time_start = time.time()
            if was_updated:
                # save json tree
                with open(self.summary_tree_path, "w", encoding="utf-8") as f:
                    json.dump(tree, f, indent=2, ensure_ascii=False)
                print(f"[Indexer] Summary tree saved to {self.summary_tree_path}")

                with open(self.file_hashes_path, "w", encoding="utf-8") as f:
                    json.dump(hash_file, f, indent=2, ensure_ascii=False)
                print(f"[Indexer] Status file saved to {self.file_hashes_path}")

                if new_chunked_docs:
                    print(f"[Indexer] Saving {len(new_chunked_docs)} embeddings to chroma")
                    
                    batch_size = 50
                    batches = [
                        new_chunked_docs[i : i + batch_size] 
                        for i in range(0, len(new_chunked_docs), batch_size)
                    ]

                    save_tasks = [self.vector_store.aadd_documents(batch) for batch in batches]
                    await async_tqdm.gather(*save_tasks, desc="Saving to ChromaDB")
                    
                    print("[Indexer] Updated chunks saved to Chroma")
            else:
                print("[Indexer] No change detected")
            time_elapsed = time.time() - time_start
            print(f"[Indexer] Updated the DB and the jsons {time_elapsed:.2f}s")

        return tree

