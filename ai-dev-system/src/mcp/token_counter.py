"""Token Counter — measures token burn per agent call.

Wraps any LangGraph node and logs:
  - Input tokens (prompt + tool schemas)
  - Output tokens (LLM response)
  - Total tokens per call
  - Cumulative tokens across the full pipeline run
"""
import time
from functools import wraps
from typing import Callable

from langchain_core.messages import BaseMessage
from src.state import GraphState


# ── Global registry ───────────────────────────────────────────────────────────
_token_log: list[dict] = []


def get_token_report() -> str:
    """Return a formatted token usage report for the full pipeline run."""
    if not _token_log:
        return "No token data recorded yet."

    lines = [
        "\n╔══════════════════════════════════════════════════════╗",
        "║           TOKEN BURN REPORT                          ║",
        "╠══════════════════════════════════════════════════════╣",
    ]

    total_input  = 0
    total_output = 0
    total_tokens = 0

    for entry in _token_log:
        lines.append(
            f"║  [{entry['agent']:<20}]  "
            f"in={entry['input_tokens']:<6} "
            f"out={entry['output_tokens']:<6} "
            f"total={entry['total_tokens']:<6} "
            f"time={entry['duration_ms']}ms"
        )
        total_input  += entry["input_tokens"]
        total_output += entry["output_tokens"]
        total_tokens += entry["total_tokens"]

    lines += [
        "╠══════════════════════════════════════════════════════╣",
        f"║  {'TOTAL':<22}  "
        f"in={total_input:<6} "
        f"out={total_output:<6} "
        f"total={total_tokens:<6}",
        "╚══════════════════════════════════════════════════════╝\n",
    ]
    return "\n".join(lines)


def reset_token_log():
    """Clear the token log — call before each pipeline run."""
    global _token_log
    _token_log = []


def _extract_usage(response) -> tuple[int, int]:
    """
    Extract input/output token counts from an LLM response.
    Handles both OpenAI-style and generic response formats.
    """
    # LangChain ChatGeneration with usage_metadata
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        return (
            meta.get("input_tokens", 0),
            meta.get("output_tokens", 0),
        )

    # OpenAI-style response_metadata
    if hasattr(response, "response_metadata"):
        meta = response.response_metadata
        usage = meta.get("token_usage", meta.get("usage", {}))
        return (
            usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            usage.get("completion_tokens", usage.get("output_tokens", 0)),
        )

    return 0, 0


class TrackedLLM:
    """
    Wraps get_llm() and intercepts every invoke() call
    to record token usage automatically.
    """

    def __init__(self, llm, agent_name: str):
        self._llm = llm
        self._agent_name = agent_name

    def bind_tools(self, tools):
        """Pass through bind_tools and keep tracking."""
        bound = self._llm.bind_tools(tools)
        return TrackedLLM(bound, self._agent_name)

    def invoke(self, messages, **kwargs):
        start = time.time()
        response = self._llm.invoke(messages, **kwargs)
        duration_ms = int((time.time() - start) * 1000)

        input_tokens, output_tokens = _extract_usage(response)
        total = input_tokens + output_tokens

        _token_log.append({
            "agent":         self._agent_name,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  total,
            "duration_ms":   duration_ms,
        })

        print(
            f"  [TOKEN] {self._agent_name} → "
            f"in={input_tokens} out={output_tokens} "
            f"total={total} ({duration_ms}ms)"
        )
        return response

    async def ainvoke(self, messages, **kwargs):
        start = time.time()
        response = await self._llm.ainvoke(messages, **kwargs)
        duration_ms = int((time.time() - start) * 1000)

        input_tokens, output_tokens = _extract_usage(response)
        total = input_tokens + output_tokens

        _token_log.append({
            "agent":         self._agent_name,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  total,
            "duration_ms":   duration_ms,
        })

        print(
            f"  [TOKEN] {self._agent_name} → "
            f"in={input_tokens} out={output_tokens} "
            f"total={total} ({duration_ms}ms)"
        )
        return response


def get_tracked_llm(agent_name: str):
    """
    Drop-in replacement for get_llm() that tracks token usage.

    Usage:
        # Before:
        llm = get_llm()

        # After:
        llm = get_tracked_llm("Coding Agent")
    """
    from src.config.llm import get_llm
    return TrackedLLM(get_llm(), agent_name)