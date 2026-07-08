"""
src/utils/llm_logger.py

Role:
    LangChain BaseCallbackHandler that logs all LLM calls to a JSONL file.
    Captures: timestamp, model, prompt snippet, response snippet, token usage, latency.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class LLMLogger(BaseCallbackHandler):
    """Logs every LLM start/end event to a rotating JSONL file."""

    def __init__(self, log_dir: str = "data/logs", max_file_mb: int = 10):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_bytes = max_file_mb * 1024 * 1024
        self._current_path = self._latest_log()
        self._start_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------
    def _latest_log(self) -> Path:
        """Find the most recent log file, or create a new one."""
        existing = sorted(self.log_dir.glob("llm_log_*.jsonl"))
        if existing:
            return existing[-1]
        return self._new_log()

    def _new_log(self) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"llm_log_{ts}.jsonl"

    def _rotate_if_needed(self):
        if self._current_path.exists() and self._current_path.stat().st_size > self.max_bytes:
            self._current_path = self._new_log()

    def _append(self, record: dict):
        self._rotate_if_needed()
        with open(self._current_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # LangChain callback hooks
    # ------------------------------------------------------------------
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *,
        run_id: str, parent_run_id: str | None = None,
        tags: list[str] | None = None, metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.time()

        # Extract model name if possible
        model = ""
        if "id" in serialized:
            model = serialized["id"]
        elif "name" in serialized:
            model = serialized["name"]
        kwargs_model = serialized.get("kwargs", {}).get("model", "")
        if kwargs_model:
            model = kwargs_model

        # Truncate prompt to avoid huge log lines
        prompt_text = prompts[0] if prompts else ""
        prompt_preview = prompt_text[:300] + "..." if len(prompt_text) > 300 else prompt_text

        self._append({
            "event": "llm_start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "model": model,
            "prompt_preview": prompt_preview,
            "prompt_total_chars": len(prompt_text),
        })

    def on_llm_end(self, response, *, run_id: str, **kwargs: Any) -> None:
        elapsed = time.time() - self._start_times.pop(run_id, 0)

        # Extract token usage from the response
        usage = {}
        try:
            if hasattr(response, "llm_output") and response.llm_output:
                usage = response.llm_output.get("token_usage", {})
            elif hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("token_usage", {})
        except Exception:
            pass

        # Extract generated text
        text = ""
        try:
            generations = getattr(response, "generations", [[]])
            if generations and generations[0]:
                text = generations[0][0].text if hasattr(generations[0][0], "text") else str(generations[0][0])
            elif hasattr(response, "content"):
                text = response.content
        except Exception:
            text = str(response)[:500]

        # Truncate for readability
        text_preview = text[:500] + "..." if len(str(text)) > 500 else str(text)

        self._append({
            "event": "llm_end",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "latency_s": round(elapsed, 3),
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0),
            "response_preview": text_preview,
        })

    def on_llm_error(self, error: BaseException, *, run_id: str, **kwargs: Any) -> None:
        elapsed = time.time() - self._start_times.pop(run_id, 0)
        self._append({
            "event": "llm_error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "latency_s": round(elapsed, 3),
            "error": str(error)[:500],
        })
