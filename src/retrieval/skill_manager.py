"""
src/retrieval/skill_manager.py

Role:
    Loads skill .md files from skills/, indexes them into a Chroma vector collection,
    and retrieves relevant skills by semantic similarity to a query.
"""

import os
import re
import yaml
from pathlib import Path
from typing import List


class SkillManager:
    def __init__(self, skills_dir: str, embeddings, chroma_path: str):
        self.skills_dir = Path(skills_dir)
        self.embeddings = embeddings
        self.chroma_path = chroma_path
        self.collection_name = "visionAgentSkills"
        self.collection = None
        self.skill_texts: dict[str, str] = {}  # name -> full text

    # ------------------------------------------------------------------
    # Load & Index
    # ------------------------------------------------------------------
    def build_index(self) -> None:
        """Parse .md files in skills_dir, embed them, and store in Chroma."""
        from langchain_chroma import Chroma

        if not self.skills_dir.exists():
            print(f"[SkillManager] Skills dir not found: {self.skills_dir}, skipping.")
            return

        docs = []
        ids = []

        for md_file in sorted(self.skills_dir.glob("*.md")):
            name, content = self._parse_skill(md_file)
            if not content.strip():
                continue

            self.skill_texts[name] = content
            docs.append(content)
            ids.append(name)
            print(f"[SkillManager] Loaded skill: {name}")

        if not docs:
            print("[SkillManager] No skills found, skipping index.")
            return

        self.collection = Chroma.from_texts(
            texts=docs,
            embedding=self.embeddings,
            ids=ids,
            collection_name=self.collection_name,
            persist_directory=self.chroma_path,
        )
        print(f"[SkillManager] Indexed {len(docs)} skills into '{self.collection_name}'")

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    async def retrieve(self, query: str, k: int = 1) -> str:
        """Return concatenated top-k skill texts relevant to the query."""
        if self.collection is None:
            return ""

        try:
            results = self.collection.similarity_search(query, k=k)
            texts = [doc.page_content for doc in results]
            return "\n\n---\n\n".join(texts)
        except Exception as e:
            print(f"[SkillManager] Retrieval error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_skill(self, filepath: Path) -> tuple[str, str]:
        """Parse a skill .md file. Returns (name, full_text)."""
        text = filepath.read_text(encoding="utf-8")

        # Try to strip YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                except yaml.YAMLError:
                    frontmatter = {}
                body = parts[2].strip()
            else:
                body = text
        else:
            body = text

        name = filepath.stem
        # Include filename as context
        full = f"Skill: {name}\n{body}"
        return name, full
