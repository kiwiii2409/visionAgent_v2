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
import re
import json
from typing import Literal
from pathlib import Path
from collections import defaultdict

from langgraph.graph import StateGraph, START, END

from src.agents.template.schema import SearchState, EvaluationSchema, FileSelectionSchema, FinalAnswerSchema, WebSelectionSchema, QueryPlanSchema
from src.agents.template.prompts import get_evaluation_prompt, get_file_selection_prompt, get_synthesis_prompt, get_web_selection_prompt, get_query_plan_prompt


class SearchGraphBuilder:
    def __init__(self, llm, vectorstore, reranker, mcp_tools, summary_tree_path: str, max_iterations: int = 3, retrieval_k: int = 4):
        self.llm = llm
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.mcp_tools_dict = {tool.name: tool for tool in mcp_tools}

        self.max_iterations = max_iterations
        self.retrieval_k = retrieval_k

        self.tree_path = Path(summary_tree_path)

        # -- tree cache + flat file index (built once on init) --
        self._tree_cache = None
        self._path_to_summary = {}
        self._name_to_paths = defaultdict(list)
        self._build_file_index()

    # ── tree cache + flat file index ──────────────────────────────────

    def _load_tree(self) -> dict:
        """Lazy-load summary_tree.json once, cache in memory."""
        if self._tree_cache is None:
            with open(self.tree_path, "r", encoding="utf-8") as f:
                self._tree_cache = json.load(f)
            print("[Search Graph] Summary tree loaded and cached.")
        return self._tree_cache

    def _build_file_index(self):
        """Walk the summary tree to build path→summary and name→paths maps."""
        if not self.tree_path.exists():
            print("[Search Graph] WARNING: No summary_tree found, file index empty.")
            return

        tree = self._load_tree()

        def walk(node, base):
            for key, val in node.items():
                cur = str(Path(base) / key)
                if val.get("_type") == "file":
                    self._path_to_summary[cur] = val.get("summary", "")
                    self._name_to_paths[key.lower()].append(cur)
                elif val.get("_type") in ("folder", "root"):
                    walk(val.get("children", {}), cur)

        for root_str, root_node in tree.items():
            walk(root_node.get("children", {}), root_str)

        print(f"[Search Graph] File index built: {len(self._path_to_summary)} files indexed.")

    def _resolve_file(self, hint: str) -> list[str]:
        """Resolve a file name or path fragment to absolute paths (candidates)."""
        # 1. exact filename match
        name = Path(hint).name.lower()
        if name in self._name_to_paths:
            return self._name_to_paths[name]

        # 2. substring fuzzy match
        matches = [p for p in self._path_to_summary if hint.lower() in p.lower()]
        return matches[:5]

    # ── directory map helpers ────────────────────────────────────────

    def _format_subtree_to_md(self, node: dict, base_path: str, collected_summaries: dict, indent_level: int = 0, max_depth: int = 2) -> str:
        """
        Formats json tree into markdown, max_depth to limit the depth of the tree.
        """
        lines = []
        indent = "  " * indent_level

        for key, value in node.items():
            current_path = str(Path(base_path) / key)

            if value["_type"] == "file":
                summary = value.get('summary', '')
                lines.append(f"{indent}├── {key} - {summary}")
                # store summaries in dict for fast lookup later
                collected_summaries[current_path] = summary

            elif value["_type"] == "folder":
                lines.append(f"{indent}├── {key}/")

                # check whether children should still be displayed
                if indent_level < max_depth:
                    child_str = self._format_subtree_to_md(
                        value["children"], current_path, collected_summaries, indent_level + 1, max_depth)
                    if child_str:
                        lines.append(child_str)
                    else:
                        lines.append(f"{indent}  └── ... (deeper files omitted)")
                else:
                    # at max depth — show placeholder so LLM knows it continues
                    lines.append(f"{indent}  └── ... (deeper files omitted)")

        return "\n".join(lines)

    def _get_surrounding_context(self, source_str: str, explored_subtrees: set, collected_summaries: dict, max_depth: int = 2) -> list:
        """
            Finds grandparent of source_str and returns file_summaries from surrounding files starting from grandparent down to max_depth as list.
        """

        summary_tree = self._load_tree()  # uses cache, reads once
        tree_context = []
        source = Path(source_str)

        for root_str, root_node in summary_tree.items():
            # Component-aware root matching (avoids prefix false positives)
            try:
                source.relative_to(root_str)
            except ValueError:
                continue

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
                # iterate to starting point (up to grandparent)
                for part in base_parts:
                    current_node = current_node[part]["children"]

                # Case: base_parts = () -> add artifical root name
                abs_dir_path = str(Path(root_str).joinpath(*base_parts))
                if abs_dir_path not in explored_subtrees:
                    # build the md tree
                    subtree_md = self._format_subtree_to_md(
                        current_node, abs_dir_path, collected_summaries, max_depth=max_depth)

                    formatted_tree = (
                        f"### DIRECTORY MAP: {abs_dir_path}\n"
                        f"```text\n{subtree_md}\n```\n"
                    )
                    tree_context.append(formatted_tree)
                    explored_subtrees.add(abs_dir_path)
            except (KeyError, ValueError):
                print("[Search Graph] Fetching surrounding summaries failed!")
                pass

            break
        return tree_context

    async def initial_retrieval(self, state: SearchState):
        """Step 1: Fetch context"""
        print("[Search Graph] Initial Retrieval ")
        docs = await self.vectorstore.amax_marginal_relevance_search(
            state["query"],
            k=20,
            fetch_k=50,
            lambda_mult=0.5
        )

        pairs = [(state["query"], doc.page_content) for doc in docs]

        scores = self.reranker.predict(pairs)
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in scored_docs[:self.retrieval_k]]

        context = []
        tree_context = []
        paths = set()
        explored_subtrees = set()
        collected_summaries = state.get(
            "file_summaries", {})  

        for doc in top_docs:
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

            new_tree_blocks = self._get_surrounding_context(
                source_str, explored_subtrees, collected_summaries, max_depth=2)
            tree_context.extend(new_tree_blocks)

        # # merge context blocks so it's [retrieved_chunk_1, retrieved_chunk_2, ..., file_summaries_1, file_summaries_2]
        # context.extend(tree_context)
        return {
            "context_blocks": context,
            "tree_blocks": tree_context, 
            "known_file_paths": list(paths),
            "explored_subtrees": explored_subtrees,
            "file_summaries": collected_summaries

        }

    async def evaluate_context(self, state: SearchState):
        """Step 2: Decide whether we have sufficient information to answer"""
        print("[Search Graph] Evaluating Context")

        evaluator = get_evaluation_prompt() | self.llm.with_structured_output(EvaluationSchema)
        full_context = "\n\n".join(state["context_blocks"] + state.get("tree_blocks", []))
        input_data = {
            "query": state["query"],
            "context": full_context
        }
        # print(full_context)
        print(
            f"[Search Graph] Approx. tokens passed to LLM (evaluate_context): {len(str(input_data)) // 4}")

        result = await evaluator.ainvoke(input_data)
        print(
            f"[Search Graph] {'Sufficient' if result.is_sufficient else 'Insufficient'} Context : {result.reasoning}")
        return {
            "is_sufficient_flag": result.is_sufficient,
            "needs_websearch_flag": result.needs_websearch
        }

    async def explore_additional_files(self, state: SearchState):
        """Step 3: Expand search to select files and their neighboring directories/ files"""
        if state.get("needs_websearch_flag") and state.get("use_websearch"): # TODO remove local context to avoid passing again and again to model
            print("[Search Graph] Gather addtional context from the web")
            exploratory_search_tool = self.mcp_tools_dict.get(
                "exploratory_search_tool")
            read_website_tool = self.mcp_tools_dict.get("read_website_tool")
            try:
                # ddg query to return top 5 websites + snippets
                search_results = await exploratory_search_tool.ainvoke({"query": state["query"]})
                formatted_result_string = "### Retrieved Websites:"
                web_summaries = state.get("web_summaries", {})
                for idx, res in enumerate(search_results, start=1):
                    url = res.get("source", "")
                    if url:
                        web_summaries[url] = {
                            "title": res.get("title", "No Title"),
                            "snippet": res.get("snippet", "No Snippet")
                        }
                        # Format string for the LLM to read
                        formatted_result_string += f"\n{idx}. {res.get('title')}\nURL: {url}\nSnippet: {res.get('snippet')}\n"


                # let the LLM decide on <= 2 relevant urls to read fully
                url_selector = get_web_selection_prompt() | self.llm.with_structured_output(WebSelectionSchema)
                selection_input = {
                    "query": state["query"],
                    "search_results": formatted_result_string
                }

                url_response = await url_selector.ainvoke(selection_input)
                if url_response.selected_urls is None:
                    new_context = [
                        f"### WEB SEARCH SNIPPETS\n```text\n{search_results}\n```\n"
                    ]
                else:
                    new_context = []  # no reason to keep snippets as context around if i read the whole file
                    for url in url_response.selected_urls:
                        try:
                            print(f"[Search Graph] Reading website: {url}")
                            web_content = await read_website_tool.ainvoke({"url": url})
                            formatted_web = (
                                f"### FULL WEBSITE CONTENT: {url}\n"
                                f"```\n{web_content}\n```\n"
                            )
                            new_context.append(formatted_web)
                        except Exception as e:
                            new_context.append(
                                f"> ERROR READING WEBSITE {url}: {e}")

                return {
                    "context_blocks": state["context_blocks"] + new_context,
                    "iterations": state.get("iterations", 0) + 1,
                    "needs_websearch_flag": False,  # reset flag so webserach is reevaluated for next step
                    "web_summaries": web_summaries 
                }
            except Exception as e:
                print(f"[Search Graph] Web search execution failed: {e}")
                return {
                    "iterations": state.get("iterations", 0) + 1,
                    "needs_websearch_flag": False
                }
        else:
            print("[Search Graph] Gather additional local context")

            file_selector = get_file_selection_prompt(
            ) | self.llm.with_structured_output(FileSelectionSchema)
            # print("*" * 50)
            # print("\n".join(state["context_blocks"]) )
            # print("*" * 50)
            # selects up to 3 relevant files using summaries of surrounding files
            full_context = "\n\n".join(state["context_blocks"] + state.get("tree_blocks", []))
            input_data = {
                "query": state["query"],
                "known_files": state["known_file_paths"],
                "context": full_context
            }


            print(
                    f"[Search Graph] Approx. tokens passed to LLM (explore_additional_files): {len(str(input_data)) // 4}")

            # selects up to 3 relevant files using summaries of surrounding files
            files_response = await file_selector.ainvoke(input_data)
            read_tool = self.mcp_tools_dict.get("read_document_tool")
            new_context = []
            new_tree_context = []
            new_paths = []

            explored_subtrees = state.get("explored_subtrees", set())
            collected_summaries = state.get("file_summaries", {})

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
                    new_tree_blocks = self._get_surrounding_context(
                        file_path, explored_subtrees, collected_summaries, max_depth=2)
                    new_tree_context.extend(new_tree_blocks)
                except Exception as e:
                    new_context.append(f"> ERROR READING {file_path}: {e}")


            print(
                f"[Search Graph] Fetching additional context from: {new_paths}")
            return {
                "context_blocks": state["context_blocks"] + new_context,
                "tree_blocks": state["tree_blocks"] + new_tree_context,
                "known_file_paths": state["known_file_paths"] + new_paths,
                "explored_subtrees": explored_subtrees,
                "iterations": state.get("iterations", 0) + 1,
                "file_summaries": collected_summaries
            }

    async def synthesize_answer(self, state: SearchState):
        """Step 4: Generate final answer"""
        print(f"[Search Graph] Synthesizing Final Answer")

        final_answer = get_synthesis_prompt() | self.llm.with_structured_output(FinalAnswerSchema)
        full_context = "\n\n".join(state["context_blocks"] + state.get("tree_blocks", []))
        input_data = {
            "query": state["query"],
            "context": full_context
        }


        print(
                f"[Search Graph] Approx. tokens passed to LLM (synthesize_answer): {len(str(input_data)) // 4}")

        response = await final_answer.ainvoke(input_data)
        file_summaries = state.get("file_summaries", {})
        web_summaries = state.get("web_summaries", {}) 

        # gather additional information besdies path for google-like overview
        enriched_sources = []
        raw_sources = response.sources if hasattr(
            response, 'sources') and response.sources else []

        for src in raw_sources:
            path_str = src if isinstance(src, str) else src.get(
                "source", src.get("path", ""))
            if not path_str:
                continue

            if path_str.startswith("http://") or path_str.startswith("https://"):
                web_meta = web_summaries.get(path_str, {})
                enriched_sources.append({
                    "name": web_meta.get("title", path_str), 
                    "path": path_str,
                    "summary": web_meta.get("snippet", "Web search result.")
                })
            else:
                file_name = Path(path_str).name
                # lookup: tree-collected summaries first, then flat index, then fallback
                file_summary = (file_summaries.get(path_str) or
                                self._path_to_summary.get(path_str, "No summary available."))

                enriched_sources.append({
                    "name": file_name,
                    "path": path_str,
                    "summary": file_summary
                })

        return {
            "final_answer": response.answer,
            "sources": enriched_sources
        }

    # ── intent routing: plan_query + targeted_read + structural_overview ──

    async def plan_query(self, state: SearchState):
        """Classify intent: targeted_file / structural_overview / broad_semantic."""
        planner = get_query_plan_prompt() | self.llm.with_structured_output(QueryPlanSchema)
        result = await planner.ainvoke({"query": state["query"]})
        hints = list(result.target_hints) if result.target_hints else []

        # Fallback: if LLM classified as targeted_file but returned no hints,
        # extract file-like tokens from the query text directly.
        if result.intent == "targeted_file" and not hints:
            query = state["query"]
            # Match patterns like "foo.py", "foo.js", "path/to/foo.py", "FooBar.hs"
            extracted = re.findall(r'[\w/.-]+\.\w{1,6}', query)
            hints = [h for h in extracted if not h.startswith(('http://', 'https://'))]
            if hints:
                print(f"[Search Graph] Intent: targeted_file | Hints (regex fallback): {hints}")
            else:
                print(f"[Search Graph] Intent: targeted_file | Hints: [] (no file patterns found)")
        else:
            print(f"[Search Graph] Intent: {result.intent} | Hints: {hints}")

        return {
            "intent": result.intent,
            "target_hints": hints
        }

    async def targeted_read(self, state: SearchState):
        """Read specific files named in the query. Falls back to vector search on miss."""
        hints = state.get("target_hints", [])
        context: list[str] = []
        tree_context: list[str] = []
        paths: list[str] = []
        explored_subtrees = state.get("explored_subtrees", set())
        collected_summaries = state.get("file_summaries", {})
        read_tool = self.mcp_tools_dict.get("read_document_tool")

        if not read_tool:
            print("[Search Graph] targeted_read: read_document_tool not available")
            return {"resolution_failed": True}

        # Resolve all hints
        all_resolved: list[str] = []
        for hint in hints:
            resolved = self._resolve_file(hint)
            all_resolved.extend(resolved)

        # Deduplicate while preserving order
        seen = set()
        all_resolved = [p for p in all_resolved if not (p in seen or seen.add(p))]

        if not all_resolved:
            print(f"[Search Graph] targeted_read: no file matched hints {hints}")
            return {"resolution_failed": True}

        # Single match → read the full file
        if len(all_resolved) == 1:
            path = all_resolved[0]
            print(f"[Search Graph] targeted_read: reading {path}")
            try:
                content = await read_tool.ainvoke({"path": path})
                context.append(f"### FULL FILE CONTENT: {path}\n```\n{content}\n```\n")
                paths.append(path)
                new_tree = self._get_surrounding_context(path, explored_subtrees, collected_summaries)
                tree_context.extend(new_tree)
            except Exception as e:
                context.append(f"> ERROR reading {path}: {e}")
        else:
            # Multiple candidates → present them for the LLM to decide
            print(f"[Search Graph] targeted_read: {len(all_resolved)} candidates")
            candidate_lines = ["### CANDIDATE FILES (multiple matches, resolve ambiguity)"]
            for p in all_resolved:
                s = self._path_to_summary.get(p, "")
                candidate_lines.append(f"- {p}: {s}")
            context.append("\n".join(candidate_lines))

        return {
            "context_blocks": context,
            "tree_blocks": tree_context,
            "known_file_paths": paths,
            "resolution_failed": False,
            "explored_subtrees": explored_subtrees,
            "file_summaries": collected_summaries
        }

    async def structural_overview_node(self, state: SearchState):
        """Generate directory maps from summary tree, no vector search."""
        hints = state.get("target_hints", [])
        tree = self._load_tree()
        tree_context: list[str] = []
        collected_summaries = state.get("file_summaries", {})

        for root_str, root_node in tree.items():
            start_node = root_node.get("children", {})
            base_path = root_str

            # If hints point to a subfolder, navigate there
            if hints:
                for hint in hints:
                    parts = Path(hint).parts
                    current = start_node
                    cur_base = base_path
                    for part in parts:
                        if part in current and current[part].get("_type") == "folder":
                            current = current[part].get("children", {})
                            cur_base = str(Path(cur_base) / part)
                    start_node = current
                    base_path = cur_base

            subtree_md = self._format_subtree_to_md(start_node, base_path, collected_summaries, max_depth=3)
            tree_context.append(
                f"### DIRECTORY MAP: {base_path}\n"
                f"```text\n{subtree_md}\n```\n"
            )

        print(f"[Search Graph] structural_overview: generated map for {base_path}")
        return {
            "context_blocks": [],  # no vector retrieval
            "tree_blocks": tree_context,
            "file_summaries": collected_summaries
        }

    # ── routing helpers ──────────────────────────────────────────────

    def _route_by_intent(self, state: dict) -> str:
        intent = state.get("intent", "broad_semantic")
        if intent == "targeted_file":
            return "targeted_read"
        elif intent == "structural_overview":
            return "structural_overview_node"
        return "initial_retrieval"

    def _route_after_targeted(self, state: dict) -> str:
        if state.get("resolution_failed"):
            return "initial_retrieval"
        return "evaluate_context"

    # ── original routing ─────────────────────────────────────────────

    def evaluation_router(self, state: dict):
        """Evaluate context & enforce max exploration iterations."""
        iterations = state.get("iterations", 0)
        max_iter = state.get("max_iterations", self.max_iterations)

        if iterations >= max_iter:
            print(
                f"[Search Graph] Max iterations ({max_iter}) reached, forcing answer synthesis")
            return "synthesize_answer"
        if state.get("is_sufficient_flag"):
            return "synthesize_answer"
        return "explore_additional_files"

    def build(self):
        workflow = StateGraph(SearchState)

        # -- new intent-routing nodes --
        workflow.add_node("plan_query", self.plan_query)
        workflow.add_node("targeted_read", self.targeted_read)
        workflow.add_node("structural_overview_node", self.structural_overview_node)

        # -- original nodes --
        workflow.add_node("initial_retrieval", self.initial_retrieval)
        workflow.add_node("evaluate_context", self.evaluate_context)
        workflow.add_node("explore_additional_files",
                          self.explore_additional_files)
        workflow.add_node("synthesize_answer", self.synthesize_answer)

        # -- new entry flow --
        workflow.add_edge(START, "plan_query")

        workflow.add_conditional_edges("plan_query", self._route_by_intent, {
            "targeted_read": "targeted_read",
            "structural_overview_node": "structural_overview_node",
            "initial_retrieval": "initial_retrieval"
        })

        workflow.add_conditional_edges("targeted_read", self._route_after_targeted, {
            "initial_retrieval": "initial_retrieval",
            "evaluate_context": "evaluate_context"
        })

        workflow.add_edge("structural_overview_node", "evaluate_context")
        workflow.add_edge("initial_retrieval", "evaluate_context")

        # -- original downstream (unchanged) --
        workflow.add_conditional_edges(
            "evaluate_context",
            self.evaluation_router
        )

        workflow.add_edge("explore_additional_files", "evaluate_context")
        workflow.add_edge("synthesize_answer", END)

        return workflow.compile()
