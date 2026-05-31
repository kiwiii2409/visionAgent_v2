"""
src/agents/search_graph.py

Role:
separate search graph as search likely wont need vlm input and can rely on hierarchical summaries etc.  and other tools to navigate file system.

Flow:
Query -> Retrieve Context -> Is Context Sufficient?     
    -> If No, take retrieved documents as starting point and use hierarchical summaries to iterate through parent/ child folders surrounding retrieved document
    -> If Yes, use retrieved context to answer query
"""

import os
import json 
from typing import Literal
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from src.agents.template.schema import SearchState, EvaluationSchema, FileSelectionSchema,FinalAnswerSchema
from src.agents.template.prompts import get_evaluation_prompt, get_file_selection_prompt, get_synthesis_prompt


class SearchGraphBuilder:
    def __init__(self, llm, vectorstore, mcp_tools_dict, summary_tree_path: str):
        self.llm = llm
        self.vectorstore = vectorstore
        self.mcp_tools = mcp_tools_dict
        self.tree_path = Path(summary_tree_path) / "tree.json"

    
    def _format_subtree_to_md(self, node: dict, indent_level: int = 0, max_depth: int = 2) -> str:
        """
        Formats json tree into markdown, max_depth to limit the depth of the tree.
        """
        lines = []
        indent = "  " * indent_level
        
        for key, value in node.items():
            if value["_type"] == "file":
                lines.append(f"{indent}├── {key} - {value.get('summary', '')}")
            elif value["_type"] == "folder":
                lines.append(f"{indent}├── {key}/")
                
                # check whether children should still be displayed
                if indent_level < max_depth:
                    child_str = self._format_subtree_to_md(value["children"], indent_level + 1, max_depth)
                    if child_str:
                        lines.append(child_str)
                else:
                    lines.append(f"{indent}  └── ... (deeper files omitted)") # placeholder s.t. llm knows that it continues there
                    
        return "\n".join(lines)

    async def initial_retrieval(self, state: SearchState):
        """Step 1: Fetch context"""
        print("[Search Graph] Initial Retrieval ")
        docs = await self.vectorstore.asimilarity_search(state["query"], k=4)
        
        context = []
        tree_context = []
        paths = set()
        added_subtrees = set()

        summary_tree = {}
        if self.tree_path.exists():
            with open(self.tree_path, "r", encoding="utf-8") as f: 
                summary_tree = json.load(f)

        for doc in docs:
            source_str = doc.metadata.get("source", "unknown_path")

            if source_str == "unknown_path":
                continue 
            
            source = Path(source_str)
            paths.add(source_str)

            context.append(f"> SOURCE: {source_str}\n{doc.page_content}")
            print(f"[Search Graph] Retrieved source:  {source_str}: {doc.page_content[:100].replace('\n', '\\n')}")            
            for root_str, root_node in summary_tree.items():
                if source_str.startswith(root_str):
                    try:
                        rel_parts = source.relative_to(root_str).parts
                        
                        # for root_str being proj/ and source being .../file.md
                        if len(rel_parts) > 2: 
                            # for proj/src/folder/file.md -> rel_parts src/folder/file.md -> return src/ 
                            base_parts = rel_parts[:-2] 
                        elif len(rel_parts) == 2: 
                            # for proj/src/file.md -> rel_parts src/file.md -> return src/ 
                            base_parts = rel_parts[:-1]
                        else: 
                            # for proj/file.md -> rel_parts file.md -> return ()
                            base_parts = ()

                        current_node = root_node["children"]
                        for part in base_parts: # iterate to starting point (up to grandparent)
                            current_node = current_node[part]["children"]
                        
                        # build the md tree
                        subtree_md = self._format_subtree_to_md(current_node, max_depth=2)
                        
                        # Case: base_parts = () -> add artifical root name
                        dir_label = '/'.join(base_parts) if base_parts else "root"
                        
                        if dir_label not in added_subtrees:
                            tree_context.append(f"> SURROUNDING FILES ({dir_label})\n{subtree_md}")
                            added_subtrees.add(dir_label)   

                    except (KeyError, ValueError):
                        pass

                    break
        
        # merge context blocks so it's [retrieved_chunk_1, retrieved_chunk_2, ..., file_summaries_1, file_summaries_2]
        context.extend(tree_context)  
        return {
            "context_blocks": context,
            "known_file_paths": list(paths),
        }
                

    async def evaluate_context(self, state: SearchState):
        """Step 2: Decide whether we have sufficient information to answer"""
        print("[Search Graph] Evaluating Context")
        
        evaluator = get_evaluation_prompt() | self.llm.with_structured_output(EvaluationSchema)
        
        result = await evaluator.ainvoke({
            "query": state["query"], 
            "context": "\n\n".join(state["context_blocks"])
        })
        
        print(f"[Search Graph] { 'Sufficient' if result.is_sufficient  else 'Insufficient'} Context : {result.reasoning}")
        return {"is_sufficient_flag": result.is_sufficient} 

    async def explore_additional_files(self, state: SearchState):
        """Step 3: Expand search to neighboring directories and read them """
        print("[Search Graph] Gather additional context")

        file_selector = get_file_selection_prompt() | self.llm.with_structured_output(FileSelectionSchema)
        
        # selects up to 3 relevant files using summaries of surrounding files
        files_response = await file_selector.ainvoke({
            "query": state["query"],
            "known_files": state["known_file_paths"],
            "context": "\n\n".join(state["context_blocks"]) 
        })

        read_tool = self.mcp_tools.get("read_file")
        new_context = []
        new_paths = []

        for file_path in files_response.selected_files:
            try:
                content = await read_tool.ainvoke({"path": file_path})
                new_context.append(f"> FULL FILE: {file_path}\n{content}")
                new_paths.append(file_path)
            except Exception as e:
                new_context.append(f"> ERROR READING {file_path}: {e}")
                
        print(f"[Search Graph] Fetching additional context from: {new_paths}")
        return {
            "context_blocks": state["context_blocks"] + new_context,
            "known_file_paths": state["known_file_paths"] + new_paths
        }


    async def synthesize_answer(self, state: SearchState):
        """Step 4: Generate final answer"""
        print(f"[Search Graph] Synthesizing Final Answer")

        chain = get_synthesis_prompt() | self.llm.with_structured_output(FinalAnswerSchema)        
        response = await chain.ainvoke({
            "query": state["query"],
            "context": "\n\n".join(state["context_blocks"])
        })
        return {
            "final_answer": response.answer,
            "sources": response.sources
        }
    
    def evaluation_router(self, state: dict):
        """Evalutates bool flag from context evaluation"""
        if state.get("is_sufficient_flag"):
            return "synthesize_answer"
        return "explore_additional_files"

    def build(self):
        workflow = StateGraph(SearchState)

        workflow.add_node("initial_retrieval", self.initial_retrieval)
        workflow.add_node("evaluate_context", self.evaluate_context)
        workflow.add_node("explore_additional_files", self.explore_additional_files)
        workflow.add_node("synthesize_answer", self.synthesize_answer)

        workflow.add_edge(START, "initial_retrieval")
        workflow.add_edge("initial_retrieval", "evaluate_context")
        
        workflow.add_conditional_edges(
            "evaluate_context", 
            self.evaluation_router
        )
        
        workflow.add_edge("explore_additional_files","synthesize_answer")
        workflow.add_edge("synthesize_answer", END)

        return workflow.compile()