import asyncio
from pathlib import Path
from typing import List, Tuple, Optional

import aiofiles
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader
from tqdm.asyncio import tqdm as async_tqdm

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    print("\n[Warning] 'markitdown' library is not installed.")



TEXT_EXTENSIONS = {".py", ".js", ".ts", ".html", ".css", ".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml", ".sh"}
RICH_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}
EXCLUDED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".xopp", ".pyc"}
EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", ".venv"}


class AsyncFileLoader:
    def __init__(self, concurreny_limit: int = 20):
        self.md = MarkItDown() # light weight library by microsoft to convert RICH_EXTENSIONS to markdown
        self.io_semaphore = asyncio.Semaphore(concurreny_limit)
        

    def _discover_files(self, folders: List[str]) -> List[Tuple[str, Path]]:
        """recursively scans folders and returns list of tuples (root_str, file_path)"""
        all_file_paths = []
        for root_str in folders:
            root = Path(root_str).resolve()
            if not root.exists():
                continue

            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                
                if any(ex_dir in file_path.parts for ex_dir in EXCLUDED_DIRS):
                    continue
                if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
                    continue
                    
                all_file_paths.append((root_str, file_path))
            
        print(f"[FileLoader] Discovered {len(all_file_paths)} files in {len(folders)} folders")
        return all_file_paths
    
    async def load_single_file(self, file_path: Path) -> Optional[Document]:
        ext = file_path.suffix.lower()
        file_path_str = str(file_path)

        async with self.io_semaphore:
            if ext in TEXT_EXTENSIONS or ext == "":
                try:
                    async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                        content = await f.read()
                        return Document(page_content=content, metadata={"source": file_path_str})
                except Exception as e :
                    print(f"[FileLoader] Error TEXT: Reading {file_path} failed with: {e}")
                    pass 

            elif ext in RICH_EXTENSIONS:
                try:
                    result = await asyncio.to_thread(self.md.convert, file_path_str)
                    if result and result.text_content:
                        return Document(page_content=result.text_content, metadata={"source": file_path_str})
                except Exception as e:
                    print(f"[FileLoader] Error RICH: Reading {file_path} failed with: {e}")
                    pass

            try:
                print("FALLBACK")
                loader = UnstructuredFileLoader(file_path_str)
                docs = await asyncio.to_thread(loader.load)
                if docs:
                    return docs[0]
            except Exception as e:
                print(f"[FileLoader] Error Fallback: Reading {file_path} failed with: {e}")
                
        return  None
    
    async def load(self, folders_to_index: List[str]) -> List[Tuple[str, Document]]:
        """Receives a list of folder paths and returns each document as 'Document'-object """
        all_file_paths = self._discover_files(folders_to_index)
        
        if not all_file_paths:
            print("[FileLoader] No supported files were found")
            return []

        load_tasks = [self.load_single_file(fp[1]) for fp in all_file_paths]
        results = await async_tqdm.gather(*load_tasks, desc="Reading Files") # preserves order => zip is valid here
        
        final_doc_tuples = []
        for (root_str, _), doc in zip(all_file_paths, results):
            if doc is not None:
                final_doc_tuples.append((root_str, doc))
                
        return final_doc_tuples