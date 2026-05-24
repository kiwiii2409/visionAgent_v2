"""
src/retrieval/indexer.py
"""
import asyncio
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

class DocumentIndexer:
    def __init__(self, vector_store: Chroma, chunk_size: int = 500, chunk_overlap: int = 128):
        self.vectorstore = vector_store
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    async def index_path(self, root: Path | str) -> dict[str, int]:
        """ingest entire folder at path and add to collection"""
        loader = DirectoryLoader(
            str(root),
            glob="**/*",
            exclude=[".git/*", "__pycache__/*", "*.png", "*.jpg"],
            loader_cls=UnstructuredFileLoader,
            recursive=True,
            show_progress=True,
            use_multithreading=True
        )
        
        docs = await asyncio.to_thread(loader.load)
        if not docs:
            return {"files_scanned": 0, "chunks_indexed": 0}

        chunks = self.text_splitter.split_documents(docs)
        await asyncio.to_thread(self.vectorstore.add_documents, chunks)
        
        return {"files_scanned": len(docs), "chunks_indexed": len(chunks)}

    async def index_string(self, text: list[dict[str, str]] | str, source_id: str) -> int:
        """add single conversation to vector store"""
        if isinstance(text, list): 
            text = "\n".join(f"[{msg['role'].upper()}]: {msg['content']}" for msg in text)
            
        if not text.strip():
            return 0

        # del old memory chunks for this task
        await asyncio.to_thread(self.vectorstore._collection.delete, where={"source": source_id})

        doc = Document(page_content=text, metadata={"source": source_id})
        chunks = self.text_splitter.split_documents([doc])
        
        if chunks:
            await asyncio.to_thread(self.vectorstore.add_documents, chunks)
            
        return len(chunks)