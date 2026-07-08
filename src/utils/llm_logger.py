"""
src/utils/llm_logger.py

Role:
    JSONL logger for LLM interactions + a wrap_llm_with_logger() helper.
    Intercepts ainvoke directly so it survives .with_structured_output() chains.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


class LLMLogger:
    """Write-only JSONL logger for LLM interactions."""

    def __init__(self, log_dir: str = "data/logs", max_file_mb: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_file_mb * 1024 * 1024
        self._current_path = self._latest_log()
        self._current_path.touch(exist_ok=True)

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------
    def _latest_log(self) -> Path:
        existing = sorted(self.log_dir.glob("llm_log_*.jsonl"))
        return existing[-1] if existing else self._new_log()

    def _new_log(self) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"llm_log_{ts}.jsonl"

    def _rotate_if_needed(self):
        if self._current_path.stat().st_size > self.max_bytes:
            self._current_path = self._new_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_start(self, model: str, prompt_chars: int, prompt_preview: str) -> str:
        run_id = f"{time.monotonic_ns()}"
        self._rotate_if_needed()
        with open(self._current_path, "a", encoding="utf-8") as f:
            json.dump({
                "event": "llm_start",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "model": model,
                "prompt_chars": prompt_chars,
                "prompt_preview": prompt_preview,
            }, f, ensure_ascii=False)
            f.write("\n")
        return run_id

    def log_end(self, run_id: str, elapsed_s: float, input_tokens: int,
                output_tokens: int, total_tokens: int, response_preview: str):
        self._rotate_if_needed()
        with open(self._current_path, "a", encoding="utf-8") as f:
            json.dump({
                "event": "llm_end",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "latency_s": round(elapsed_s, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "response_preview": response_preview,
            }, f, ensure_ascii=False)
            f.write("\n")

    def log_error(self, run_id: str, elapsed_s: float, error: str):
        self._rotate_if_needed()
        with open(self._current_path, "a", encoding="utf-8") as f:
            json.dump({
                "event": "llm_error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "latency_s": round(elapsed_s, 3),
                "error": error,
            }, f, ensure_ascii=False)
            f.write("\n")


def wrap_llm_with_logger(llm, logger: LLMLogger, model_name: str = ""):
    """
    Wrap a LangChain chat model so every ainvoke call is logged.
    Intercepts ainvoke directly — works even with .with_structured_output().
    """
    _original_ainvoke = llm.ainvoke
    _model = model_name or getattr(llm, "model_name", "") or getattr(llm, "model", "") or type(llm).__name__

    async def _logged_ainvoke(input_, config=None, **kwargs):
        prompt_text = _extract_prompt_text(input_)
        prompt_chars = len(prompt_text)
        prompt_preview = prompt_text[:300] + "..." if prompt_chars > 300 else prompt_text
        run_id = logger.log_start(_model, prompt_chars, prompt_preview)

        t0 = time.time()
        try:
            result = await _original_ainvoke(input_, config=config, **kwargs)
            elapsed = time.time() - t0

            input_tokens = output_tokens = total_tokens = 0
            try:
                meta = getattr(result, "response_metadata", {}) or {}
                usage = meta.get("token_usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            except Exception:
                pass

            response_text = _extract_response_text(result)
            response_preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
            logger.log_end(run_id, elapsed, input_tokens, output_tokens, total_tokens, response_preview)
            return result

        except Exception as e:
            elapsed = time.time() - t0
            logger.log_error(run_id, elapsed, str(e)[:500])
            raise

    llm.ainvoke = _logged_ainvoke
    return llm


def _extract_prompt_text(input_) -> str:
    if isinstance(input_, str):
        return input_
    if isinstance(input_, list):
        parts = []
        for item in input_:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(input_, dict):
        return json.dumps(input_, ensure_ascii=False)
    return str(input_)


def _extract_response_text(result) -> str:
    content = getattr(result, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return str(result)
