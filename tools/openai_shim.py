"""openai_shim — an OpenAI-compatible HTTP server backed by the Claude Code CLI + local embeddings.

Why: Mem0, Graphiti, A-MEM, GraphRAG and Letta all speak the OpenAI protocol. This repo's LLM
budget is the Claude Code subscription (``claude -p``), not an API key. The shim lets every
third-party memory system run against the SAME model and the SAME embedding model, which is the
MemDelta requirement for a fair comparison: one generator, one embedder, only the memory changes.

Endpoints
  POST /v1/chat/completions   messages -> `claude -p --max-turns 1 --tools ""` (no agent tools)
                              supports response_format (json_object / json_schema) and emulated
                              function calling (tools -> JSON tool_calls, finish_reason=tool_calls)
  POST /v1/embeddings         sentence-transformers (default BAAI/bge-small-en-v1.5, 384-d)
  GET  /v1/models

Every call is appended to SHIM_LOG (JSONL) with latency, tokens, cost and the X-Caller header, so a
memory system's WRITE-PATH cost (ingest-time LLM calls) is measured, not estimated.

Run:  python -m tools.openai_shim --port 8008
Use:  OPENAI_BASE_URL=http://127.0.0.1:8008/v1  OPENAI_API_KEY=shim
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="openai-shim (claude -p)")
_SEM = asyncio.Semaphore(int(os.environ.get("SHIM_CONCURRENCY", "4")))
_LOG = os.environ.get("SHIM_LOG", "results/feasibility/shim_calls.jsonl")
_EMBED_MODEL = os.environ.get("SHIM_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_embedder = None
_embed_lock = threading.Lock()
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def _model_alias(name: str | None) -> str:
    n = (name or "").lower()
    if "haiku" in n:
        return "haiku"
    if "opus" in n:
        return "opus"
    return os.environ.get("SHIM_DEFAULT_MODEL", "sonnet")


def _render(messages: list[dict], tools: list[dict] | None, response_format: dict | None,
            tool_choice: Any) -> tuple[str, str]:
    """Split OpenAI messages into (system_prompt, user_prompt) for `claude -p`."""
    system_parts, transcript = [], []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):  # multimodal parts -> text only
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = content or ""
        if role == "system" or role == "developer":
            system_parts.append(content)
        elif role == "tool":
            transcript.append(f"[tool result for call {m.get('tool_call_id')}]\n{content}")
        elif role == "assistant":
            if m.get("tool_calls"):
                calls = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments")}
                         for c in m["tool_calls"]]
                transcript.append(f"[assistant tool_calls]\n{json.dumps(calls)}")
            if content:
                transcript.append(f"[assistant]\n{content}")
        else:
            transcript.append(f"[user]\n{content}")
    system = "\n\n".join(system_parts) or "You are a helpful assistant."
    system = ("You are a plain chat-completion backend. Answer the final user turn directly. "
              "No preamble, no tool use of your own.\n\n" + system)
    if tools:
        spec = json.dumps([t.get("function", t) for t in tools], indent=1)
        forced = ""
        if isinstance(tool_choice, dict):
            forced = f" You MUST call the function named {tool_choice.get('function', {}).get('name')!r}."
        elif tool_choice == "required":
            forced = " You MUST call at least one function."
        system += ("\n\nFUNCTION CALLING: you may call these functions:\n" + spec +
                   "\nTo call functions, reply with ONLY a JSON object of the form "
                   '{"tool_calls":[{"name":"<fn>","arguments":{...}}]} and nothing else. '
                   "Otherwise reply normally." + forced)
    if response_format and response_format.get("type") in ("json_object", "json_schema"):
        system += "\n\nOUTPUT: reply with a single valid JSON value only — no markdown fences, no commentary."
    return system, "\n\n".join(transcript) or "(empty)"


async def _claude(system: str, prompt: str, model: str, json_schema: dict | None) -> dict:
    cmd = [shutil.which("claude") or "claude", "-p", "--model", model, "--output-format", "json",
           "--max-turns", "3" if json_schema else "1",   # structured output spends extra turns on its tool
           "--tools", "", "--no-session-persistence", "--system-prompt", system]
    if json_schema:
        cmd += ["--json-schema", json.dumps(json_schema)]
    async with _SEM:
        proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=float(os.environ.get("SHIM_TIMEOUT", "300")))
    if proc.returncode != 0:
        _log_failure({"stage": "claude", "rc": proc.returncode, "stderr": err.decode()[-1500:], "stdout": out.decode()[-800:],
                      "system_head": system[:1500], "prompt_head": prompt[:1500], "schema": json_schema, "model": model})
        raise RuntimeError(f"claude exited {proc.returncode}: {err.decode()[-400:] or out.decode()[-200:]}")
    data = json.loads(out.decode())
    if data.get("is_error") or data.get("subtype") not in (None, "success"):
        raise RuntimeError(f"claude error: {str(data.get('result'))[:400]}")
    return data


def _parse_tool_calls(text: str) -> list[dict] | None:
    t = _FENCE.sub("", text.strip())
    if not t.startswith("{"):
        return None
    try:
        obj = json.loads(t)
    except ValueError:
        return None
    calls = obj.get("tool_calls") if isinstance(obj, dict) else None
    if not calls:
        return None
    return [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
             "function": {"name": c.get("name"), "arguments": json.dumps(c.get("arguments", {}))}} for c in calls]


def _log_failure(rec: dict) -> None:
    try:
        path = os.path.splitext(_LOG)[0] + "_failures.jsonl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _log(rec: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_LOG), exist_ok=True)
        with open(_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


@app.post("/v1/chat/completions")
async def chat(request: Request, x_caller: str | None = Header(default=None)):
    body = await request.json()
    model = _model_alias(body.get("model"))
    rf = body.get("response_format")
    schema = rf.get("json_schema", {}).get("schema") if rf and rf.get("type") == "json_schema" else None
    system, prompt = _render(body.get("messages", []), body.get("tools"), rf, body.get("tool_choice"))
    t0 = time.time()
    try:
        data = await _claude(system, prompt, model, schema)
    except Exception as e:  # noqa: BLE001
        _log({"ts": t0, "caller": x_caller, "model": model, "error": str(e)[:300]})
        return JSONResponse({"error": {"message": str(e), "type": "shim_error"}}, status_code=500)
    text = data.get("structured_output")
    text = json.dumps(text) if isinstance(text, (dict, list)) else (data.get("result") or "")
    if rf and rf.get("type") in ("json_object", "json_schema"):
        text = _FENCE.sub("", text.strip())
    u = data.get("usage", {})
    prompt_tokens = int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0))
    completion_tokens = int(u.get("output_tokens", 0))
    tool_calls = _parse_tool_calls(text) if body.get("tools") else None
    message = {"role": "assistant", "content": None if tool_calls else text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion", "created": int(t0),
        "model": body.get("model") or model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }
    _log({"ts": t0, "caller": x_caller, "model": model, "latency_s": round(time.time() - t0, 3),
          "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
          "cost_usd": data.get("total_cost_usd"), "tool_calls": len(tool_calls or []),
          "json": bool(rf), "chars_in": len(prompt) + len(system)})
    if body.get("stream"):
        async def gen():
            chunk = {"id": resp["id"], "object": "chat.completion.chunk", "created": resp["created"], "model": resp["model"],
                     "choices": [{"index": 0, "delta": message, "finish_reason": resp["choices"][0]["finish_reason"]}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(resp)


def _get_embedder():
    global _embedder
    with _embed_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


@app.post("/v1/embeddings")
async def embeddings(request: Request, x_caller: str | None = Header(default=None)):
    body = await request.json()
    inp = body.get("input")
    texts = [inp] if isinstance(inp, str) else list(inp or [])
    t0 = time.time()
    vecs = await asyncio.get_event_loop().run_in_executor(None, lambda: _get_embedder().encode(texts, normalize_embeddings=True))
    _log({"ts": t0, "caller": x_caller, "embed": len(texts), "latency_s": round(time.time() - t0, 3)})
    return JSONResponse({
        "object": "list", "model": _EMBED_MODEL,
        "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
        "usage": {"prompt_tokens": sum(len(t) // 4 for t in texts), "total_tokens": sum(len(t) // 4 for t in texts)},
    })


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "shim"}
                                       for m in ("sonnet", "haiku", "opus", "gpt-4o", "gpt-4o-mini", "text-embedding-3-small", _EMBED_MODEL)]}


@app.get("/health")
async def health():
    return {"ok": True, "embed_model": _EMBED_MODEL, "log": _LOG}


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
