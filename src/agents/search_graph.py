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
    def __init__(self, llm, vectorstore, mcp_tools, summary_tree_path: str, max_iterations: int = 3, retrieval_k: int = 4):
        self.llm = llm
        self.vectorstore = vectorstore
        self.mcp_tools_dict = {tool.name: tool for tool in  mcp_tools}

        self.max_iterations = max_iterations
        self.retrieval_k = retrieval_k

        self.tree_path = Path(summary_tree_path)

    
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
    
    def _get_surrounding_context(self, source_str: str, explored_subtrees: set, max_depth: int =2) -> list:
        """
            Finds grandparent of source_str and returns file_summaries from surrounding files starting from grandparent down to max_depth as list.
        """

        if self.tree_path.exists():
            with open(self.tree_path, "r", encoding="utf-8") as f: 
                self.summary_tree = json.load(f)
        else: 
            print("[Search Graph] ERROR: No summary_tree found!")

        tree_context = []
        source = Path(source_str)

        for root_str, root_node in self.summary_tree.items():
            if source_str.startswith(root_str):
                try:
                    rel_parts = source.relative_to(root_str).parts
                    
                    # for root_str being proj/... and source being .../file.md
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
                    subtree_md = self._format_subtree_to_md(current_node, max_depth=max_depth)
                    
                    # Case: base_parts = () -> add artifical root name
                    abs_dir_path = str(Path(root_str).joinpath(*base_parts))
                    if abs_dir_path not in explored_subtrees:
                        formatted_tree = (
                            f"### DIRECTORY MAP: {abs_dir_path}\n"
                            f"```text\n{subtree_md}\n```\n"
                        )
                        tree_context.append(formatted_tree)                        
                        explored_subtrees.add(abs_dir_path)   

                except (KeyError, ValueError):
                    pass

                break
        

        return tree_context

    async def initial_retrieval(self, state: SearchState):
        """Step 1: Fetch context"""
        print("[Search Graph] Initial Retrieval ")
        docs = await self.vectorstore.amax_marginal_relevance_search(state["query"], k=self.retrieval_k, fetch_k=20, lambda_mult=0.5)
        
        context = []
        tree_context = []
        paths = set()
        explored_subtrees = set()

        for doc in docs:
            source_str = doc.metadata.get("source", "unknown_path")

            if source_str == "unknown_path":
                continue 
            
            paths.add(source_str)

            formatted_chunk = (
                f"### RETRIEVED SNIPPET: {source_str}\n"
                f"```text\n{doc.page_content}\n```\n"
            )
            context.append(formatted_chunk)
            preview = doc.page_content[:100].replace('\n', '\\n')
            print(f"[Search Graph] Retrieved source:  {source_str}: {preview}")

            new_tree_blocks = self._get_surrounding_context(source_str, explored_subtrees, max_depth= 2    )
            tree_context.extend(new_tree_blocks)

        # merge context blocks so it's [retrieved_chunk_1, retrieved_chunk_2, ..., file_summaries_1, file_summaries_2]
        context.extend(tree_context)  

        return {
            "context_blocks": context,
            "known_file_paths": list(paths),
            "explored_subtrees": explored_subtrees
        }
                

    async def evaluate_context(self, state: SearchState):
        """Step 2: Decide whether we have sufficient information to answer"""
        print("[Search Graph] Evaluating Context")
        
        evaluator = get_evaluation_prompt() | self.llm.with_structured_output(EvaluationSchema)
        
        input_data = {
            "query": state["query"], 
            "context": "\n\n".join(state["context_blocks"])
        }

        try:
            tokens = self.llm.get_num_tokens(get_evaluation_prompt().format(**input_data))
            print(f"[Search Graph] Tokens passed to LLM (evaluate_context): {tokens}")
        except Exception:
            print(f"[Search Graph] Approx. tokens passed to LLM (evaluate_context): {len(str(input_data)) // 4}")

        result = await evaluator.ainvoke(input_data)
        
        print(f"[Search Graph] { 'Sufficient' if result.is_sufficient  else 'Insufficient'} Context : {result.reasoning}")
        return {"is_sufficient_flag": result.is_sufficient} 

    async def explore_additional_files(self, state: SearchState):
        """Step 3: Expand search to select files and their neighboring directories/ files"""
        print("[Search Graph] Gather additional context")

        file_selector = get_file_selection_prompt() | self.llm.with_structured_output(FileSelectionSchema)
        # print("*" * 50)
        # print("\n".join(state["context_blocks"]) )
        # print("*" * 50)
        # selects up to 3 relevant files using summaries of surrounding files
        input_data = {
            "query": state["query"],
            "known_files": state["known_file_paths"],
            "context": "\n".join(state["context_blocks"]) 
        }

        try:
            tokens = self.llm.get_num_tokens(get_file_selection_prompt().format(**input_data))
            print(f"[Search Graph] Tokens passed to LLM (explore_additional_files): {tokens}")
        except Exception:
            print(f"[Search Graph] Approx. tokens passed to LLM (explore_additional_files): {len(str(input_data)) // 4}")

        # selects up to 3 relevant files using summaries of surrounding files
        files_response = await file_selector.ainvoke(input_data)
        read_tool = self.mcp_tools_dict.get("read_document_tool")
        new_context = []
        new_tree_context = []
        new_paths = []

        explored_subtrees = state.get("explored_subtrees", set())

        for file_path in files_response.selected_files:
            try:
                content = await read_tool.ainvoke({"path": file_path})
                formatted_file = (
                    f"### FULL FILE CONTENT: {file_path}\n"
                    f"```\n{content}\n```\n"
                )
                new_context.append(formatted_file)
                new_paths.append(file_path)
                
                # fetch the surrounding directory content from the selected files
                new_tree_blocks = self._get_surrounding_context(file_path, explored_subtrees, max_depth=2)
                new_tree_context.extend(new_tree_blocks)
            except Exception as e:
                new_context.append(f"> ERROR READING {file_path}: {e}")
                
        new_context.extend(new_tree_context)  


        print(f"[Search Graph] Fetching additional context from: {new_paths}")
        return {
                "context_blocks": state["context_blocks"] + new_context,
                "known_file_paths": state["known_file_paths"] + new_paths,
                "explored_subtrees": explored_subtrees,
                "iterations": state.get("iterations", 0) + 1
        }


    async def synthesize_answer(self, state: SearchState):
        """Step 4: Generate final answer"""
        print(f"[Search Graph] Synthesizing Final Answer")

        chain = get_synthesis_prompt() | self.llm.with_structured_output(FinalAnswerSchema)        
        input_data = {
            "query": state["query"],
            "context": "\n\n".join(state["context_blocks"])
        }

        try:
            tokens = self.llm.get_num_tokens(get_synthesis_prompt().format(**input_data))
            print(f"[Search Graph] Tokens passed to LLM (synthesize_answer): {tokens}")
        except Exception:
            print(f"[Search Graph] Approx. tokens passed to LLM (synthesize_answer): {len(str(input_data)) // 4}")

        response = await chain.ainvoke(input_data)
        return {
            "final_answer": response.answer,
            "sources": response.sources
        }
    
    def evaluation_router(self, state: dict):
        """Evaluate context & enforce max exploration iterations."""
        iterations = state.get("iterations", 0)
        max_iter = state.get("max_iterations", self.max_iterations)

        if iterations >= max_iter:
            print(f"[Search Graph] Max iterations ({max_iter}) reached, forcing answer synthesis")
            return "synthesize_answer"
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
        
        workflow.add_edge("explore_additional_files","evaluate_context")
        workflow.add_edge("synthesize_answer", END)

        return workflow.compile()
    


