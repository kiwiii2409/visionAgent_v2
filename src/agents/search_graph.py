"""
separate search graph as search likely wont need vlm input and can rely on hierarchical summaries etc.  and other tools to navigate file system.

Flow:
Query -> Retrieve Context -> Is Context Sufficient?     -> If No, take retrieved documents as starting point and use hierarchical summaries to iterate through parent/ child folders surrounding retrieved document
                                                        -> If Yes, use retrieved context to answer query
"""

import os
from typing import Literal
from langgraph.graph import StateGraph, START, END

from src.agents.templates.schema import SearchState, EvaluationSchema, FileSelectionSchema
from src.agents.templates.prompts import get_evaluation_prompt, get_file_selection_prompt, get_synthesis_prompt


class SearchGraphBuilder:
    def __init__(self, llm, vectorstore, mcp_tools_dict):
        self.llm = llm
        self.vectorstore = vectorstore
        self.mcp_tools = mcp_tools_dict

    async def initial_retrieval(self, state: SearchState):
        """Step 1: fetch context"""
        print("[Search Graph] Initial Retrieval ")
        docs = await self.vectorstore.asimilarity_search(state["query"], k=4)
        
        context = []
        paths = set()
        dirs = set()
        
        for doc in docs:
            source = doc.metadata.get("source", "unknown_path")
            context.append(f"--- SOURCE: {source} ---\n{doc.page_content}")
            if source != "unknown_path":
                paths.add(source)
                dirs.add(os.path.dirname(source)) 
                
        return {
            "context_blocks": context,
            "known_file_paths": list(paths),
            "directories_to_explore": list(dirs)
        }

    async def evaluate_context(self, state: SearchState):
        """Step 2: decide whether we have sufficient information to answer"""
        print("[Search Graph] Evaluating Context")
        
        evaluator = get_evaluation_prompt() | self.llm.with_structured_output(EvaluationSchema)
        
        result = await evaluator.ainvoke({
            "query": state["query"], 
            "context": "\n\n".join(state["context_blocks"])
        })
        
        print(f"   >>> Step 2 Context Evaluation: {result.is_sufficient} ({result.reasoning})")
        return {"is_sufficient_flag": result.is_sufficient} 

    async def explore_directories(self, state: SearchState):
        """Step 3: expand search to neighboring directories"""
        print("--- [Search Graph] Expanding Search Radius ---")
        list_tool = self.mcp_tools.get("list_directory")
        all_directory_contents = ""
        
        for directory in state["directories_to_explore"][:3]: 
            try:
                contents = await list_tool.ainvoke({"path": directory})
                all_directory_contents += f"\nContents of {directory}:\n{contents}\n"
            except Exception:
                continue
                
        selector = get_file_selection_prompt() | self.llm.with_structured_output(FileSelectionSchema)
        
        result = await selector.ainvoke({
            "query": state["query"],
            "known_files": state["known_file_paths"],
            "dir_contents": all_directory_contents
        })
        
        print(f"   >>> Step 3 Additional Files: {result.selected_files}")
        return {"files_to_read": result.selected_files}

    async def read_selected_files(self, state: SearchState):
        """Step 4: read new files and add to context"""
        read_tool = self.mcp_tools.get("read_file")
        new_context = []
        new_paths = []
        
        for file_path in state["files_to_read"]:
            try:
                content = await read_tool.ainvoke({"path": file_path})
                new_context.append(f"\nFile Path: {file_path} ---\n{content}")
                new_paths.append(file_path)
            except Exception as e:
                new_context.append(f"\nError reading {file_path}: {e} ---")
                
        return {
            "context_blocks": state["context_blocks"] + new_context,
            "known_file_paths": state["known_file_paths"] + new_paths
        }

    async def synthesize_answer(self, state: SearchState):
        """Step 5: generate final answer"""

        chain = get_synthesis_prompt() | self.llm
        
        response = await chain.ainvoke({
            "query": state["query"],
            "context": "\n\n".join(state["context_blocks"])
        })
        
        return {"final_answer": response.content}

    def evaluation_router(self, state: dict):
        """evalutates bool flag from context evaluation"""
        if state.get("is_sufficient_flag"):
            return "synthesize_answer"
        return "explore_directories"

    def build(self):
        workflow = StateGraph(SearchState)

        workflow.add_node("initial_retrieval", self.initial_retrieval)
        workflow.add_node("evaluate_context", self.evaluate_context)
        workflow.add_node("explore_directories", self.explore_directories)
        workflow.add_node("read_selected_files", self.read_selected_files)
        workflow.add_node("synthesize_answer", self.synthesize_answer)

        workflow.add_edge(START, "initial_retrieval")
        workflow.add_edge("initial_retrieval", "evaluate_context")
        
        workflow.add_conditional_edges(
            "evaluate_context", 
            self.evaluation_router
        )
        
        workflow.add_edge("explore_directories", "read_selected_files")
        workflow.add_edge("read_selected_files", "synthesize_answer")
        workflow.add_edge("synthesize_answer", END)

        return workflow.compile()