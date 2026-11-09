"""Shared LLM client using Claude Code CLI.

Uses the `claude` CLI in pipe mode so the experiment runs on
the user's existing Claude Code subscription — no extra API costs.

All architectures use this client for LLM calls, ensuring
consistent behavior and cost tracking.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from typing import Any

from src.evaluation.cost_tracker import APICall, CostTracker

_claude_path: str | None = None


def _find_claude() -> str:
    global _claude_path
    if _claude_path is None:
        _claude_path = shutil.which("claude")
        if not _claude_path:
            raise RuntimeError(
                "Claude Code CLI not found. Install it: https://docs.anthropic.com/en/docs/claude-code"
            )
    return _claude_path


def ensure_api_key() -> str:
    """Verify claude CLI is available (no separate API key needed)."""
    _find_claude()
    return "claude-code"


async def _call_claude(prompt: str, model: str = "sonnet") -> dict:
    """Call claude CLI in pipe mode, return parsed response.

    Args:
        prompt: The user prompt to send.
        model: "sonnet" for fast/cheap, "opus" for judge quality.
    """
    claude_bin = _find_claude()

    # Build command — use --model flag for model selection
    cmd = [claude_bin, "-p", "--output-format", "json"]
    if model == "sonnet":
        cmd.extend(["--model", "sonnet"])
    # Default (no --model flag) uses the user's configured default

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=prompt.encode())

    if proc.returncode != 0:
        raise RuntimeError(f"Claude CLI failed (exit {proc.returncode}): {stderr.decode()[:500]}")

    response = json.loads(stdout.decode())
    return response


def _extract_text(response: dict) -> str:
    """Extract the text result from claude CLI JSON response."""
    return response.get("result", "")


def _extract_usage(response: dict) -> dict:
    """Extract token usage from claude CLI response."""
    usage = response.get("usage", {})
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
        "cost_usd": response.get("total_cost_usd", 0),
    }


async def chat_completion(
    messages: list[dict[str, str]],
    model: str = "inference",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    purpose: str = "inference",
    cost_tracker: CostTracker | None = None,
    response_format: dict | None = None,
) -> str:
    """Make a Claude Code CLI call with cost tracking.

    Returns the assistant message content.
    """
    # Map model to sonnet/opus
    cli_model = "sonnet"
    if model in ("judge", "gpt-4o", "opus"):
        cli_model = "opus"

    # Build prompt from messages
    parts = []
    for msg in messages:
        if msg["role"] == "system":
            parts.append(f"[System instruction]: {msg['content']}")
        elif msg["role"] == "user":
            parts.append(msg["content"])

    prompt = "\n\n".join(parts)

    # Add JSON instruction if needed
    if response_format and response_format.get("type") == "json_object":
        prompt += "\n\nIMPORTANT: Respond with valid JSON only. No markdown fences, no explanation — just the raw JSON object."

    response = await _call_claude(prompt, model=cli_model)
    text = _extract_text(response)
    usage = _extract_usage(response)

    if cost_tracker:
        cost_tracker.record(
            APICall(
                model=f"claude-{cli_model}",
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                purpose=purpose,
                cached_input_tokens=usage["cache_read"],
            )
        )

    return text


async def chat_completion_json(
    messages: list[dict[str, str]],
    model: str = "inference",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    purpose: str = "inference",
    cost_tracker: CostTracker | None = None,
) -> dict:
    """Make a Claude Code CLI call that returns JSON."""
    text = await chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        purpose=purpose,
        cost_tracker=cost_tracker,
        response_format={"type": "json_object"},
    )

    # Clean up response — strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise
