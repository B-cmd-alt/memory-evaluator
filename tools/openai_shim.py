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
    if "sonnet" in n:
        return "sonnet"   # lets a caller pin Sonnet (the judge) while SHIM_DEFAULT_MODEL selects the generator tier
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


# Optional deterministic prompt cache (SHIM_CACHE=path.sqlite). Identical (model, system, prompt, schema) requests replay the
# stored response instead of calling the model again. Round 3 uses it so a rebuild-on-delete (Mastra OM, Generative Agents)
# re-sends the unchanged prefix of its ingest for free; every hit is logged with cached=true, so cost tables can report
# first-call (uncached) cost. Off by default: rounds 1-2 ran without it.
_CACHE_PATH = os.environ.get("SHIM_CACHE")   # a SQLite file; several shims (one per runner process) can share it


def _cache_db():
    import sqlite3
    db = sqlite3.connect(_CACHE_PATH, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
    return db


def _cache_key(system: str, prompt: str, model: str, json_schema: dict | None) -> str:
    import hashlib
    return hashlib.sha256(json.dumps([model, system, prompt, json_schema], sort_keys=True).encode()).hexdigest()


# `claude -p` adds the real calendar date to every prompt ("Today's date is ..."), which no flag removes. A reader that sees
# it treats "as of today" questions about a corpus that ends in 2025 as unanswerable (Haiku: 48% of round-4 update
# answers cited the real date; see docs/ERRATA.md). This line is appended to every system prompt on this backend, and is
# part of the cache key. SHIM_CLI_DATE_NOTE=0 reproduces the earlier behaviour.
_CLI_DATE_NOTE = ("\n\n(Note from the test harness: the software that delivers this prompt may attach the real calendar date. "
                  "That date is not part of the task. Ignore it, and do not treat it as \"today\".)") if os.environ.get("SHIM_CLI_DATE_NOTE", "1") != "0" else ""


async def _claude_cached(system: str, prompt: str, model: str, json_schema: dict | None) -> tuple[dict, bool]:
    system = (system or "") + _CLI_DATE_NOTE
    if not _CACHE_PATH:
        return await _claude(system, prompt, model, json_schema), False
    k = _cache_key(system, prompt, model, json_schema)
    db = _cache_db()
    try:
        row = db.execute("SELECT v FROM c WHERE k = ?", (k,)).fetchone()
        if row:
            return json.loads(row[0]), True
        data = await _claude(system, prompt, model, json_schema)
        db.execute("INSERT OR IGNORE INTO c VALUES (?, ?)", (k, json.dumps(data)))
        db.commit()
        return data, False
    finally:
        db.close()


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


# ── OpenRouter backend (SHIM_BACKEND=openrouter) ────────────────────────────────────────────────────────────────
# Round 4 needs several hundred thousand calls; the `claude -p` path does ~10-17 calls/min per process. With this backend
# the shim forwards OpenAI-format requests to OpenRouter with real concurrency, native tool calling and structured
# output. Model routing: a name containing "/" is an explicit OpenRouter id (the runner passes each READER this way);
# "sonnet"/"haiku"/"opus" map to the current Claude ids (the pinned judge says "sonnet"); anything else (the names
# third-party systems hard-code, e.g. "gpt-4o-mini") goes to SHIM_DEFAULT_MODEL = the WRITE-path model.
# Every upstream call's dollar cost is recorded in the shared SQLite file; SHIM_SPEND_CAP_USD is a hard stop.
_BACKEND = os.environ.get("SHIM_BACKEND", "claude_cli")
_OR_ALIAS = {"sonnet": "anthropic/claude-sonnet-5.5", "haiku": "anthropic/claude-haiku-5.5", "opus": "anthropic/claude-opus-5.5"}
_SPEND_CAP = float(os.environ.get("SHIM_SPEND_CAP_USD", "1000"))
_or_client = None


def _or_model(name: str | None) -> str:
    n = name or ""
    if "/" in n:
        return n
    d = os.environ.get("SHIM_DEFAULT_MODEL", "haiku")
    for k, v in _OR_ALIAS.items():
        if k in n.lower():
            return v
    return d if "/" in d else _OR_ALIAS.get(d, d)


def _spend_db():
    import sqlite3
    db = sqlite3.connect(_CACHE_PATH or "results/shim_spend.sqlite", timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS spend (ts REAL, model TEXT, usd REAL)")
    return db


def _mark_cacheable(messages: list[dict], model: str) -> list[dict]:
    """Provider prompt caching for long shared prefixes (the full-context baseline sends the same 100K-500K-token store
    with every question). The shared answer prompt ends '...\n\nQuestion: <q>\n\nAnswer:'; everything before the question
    is identical across questions, so it becomes its own content block with a cache breakpoint."""
    if not model.startswith(("anthropic/", "google/")) or not messages:
        return messages
    last = messages[-1]
    c = last.get("content")
    if last.get("role") != "user" or not isinstance(c, str) or len(c) < 16000:
        return messages
    q = c.rfind("\n\nQuestion: ")
    cut = c.rfind("\n", 0, max(0, q - 400)) if q > 0 else -1   # full context and full context + trailer share this prefix
    if cut <= 0:
        return messages
    blocks = [{"type": "text", "text": c[:cut], "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": c[cut:]}]
    return messages[:-1] + [{**last, "content": blocks}]


async def _openrouter(body: dict, model: str) -> tuple[dict, bool]:
    """Forward one chat completion. Returns (OpenRouter response json, served_from_shim_cache)."""
    import hashlib
    import httpx
    global _or_client
    req = {"model": model, "messages": body.get("messages", []), "temperature": 0, "usage": {"include": True}}
    for k in ("tools", "tool_choice", "response_format", "max_tokens", "stop"):
        if body.get(k) is not None:
            req[k] = body[k]
    req.setdefault("max_tokens", int(os.environ.get("SHIM_MAX_TOKENS", "16000")))   # explicit: provider defaults differ and can truncate silently
    key = hashlib.sha256(json.dumps([model, req["messages"], req.get("tools"), req.get("response_format"), req.get("tool_choice")],
                                    sort_keys=True).encode()).hexdigest()
    db = _spend_db()
    try:
        if _CACHE_PATH:
            row = db.execute("SELECT v FROM c WHERE k = ?", (key,)).fetchone()
            if row:
                return json.loads(row[0]), True
        spent = db.execute("SELECT COALESCE(SUM(usd), 0) FROM spend").fetchone()[0]
        if spent >= _SPEND_CAP:
            raise RuntimeError(f"spend cap reached: ${spent:.2f} >= ${_SPEND_CAP:.2f}")
        send = {**req, "messages": _mark_cacheable(req["messages"], model)}
        if _or_client is None:
            _or_client = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30), verify=os.environ.get("SSL_CERT_FILE") or "/root/.ccr/ca-bundle.crt")
        api_key = os.environ.get("openrouter_api") or os.environ["OPENROUTER_API_KEY"]
        last_err = None
        async with _SEM:
            for attempt in range(7):
                try:
                    r = await _or_client.post("https://openrouter.ai/api/v1/chat/completions", json=send,
                                              headers={"Authorization": f"Bearer {api_key}"})
                    if r.status_code == 400 and send.get("response_format", {}).get("type") == "json_schema":
                        # provider without schema-constrained output: fall back to JSON mode with the schema in the prompt
                        sch = send["response_format"]["json_schema"].get("schema")
                        send = {**send, "response_format": {"type": "json_object"},
                                "messages": send["messages"] + [{"role": "user", "content": "Reply with ONE JSON object matching this JSON schema, nothing else:\n" + json.dumps(sch)}]}
                        continue
                    if r.status_code in (408, 409, 425, 429) or r.status_code >= 500:
                        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                        await asyncio.sleep(min(60, 2 ** attempt))
                        continue
                    data = r.json()
                    if r.status_code != 200 or "choices" not in data:
                        err = str(data.get("error") or data)[:400]
                        if "rate" in err.lower() or "overloaded" in err.lower() or "timeout" in err.lower():
                            last_err = err
                            await asyncio.sleep(min(60, 2 ** attempt))
                            continue
                        raise RuntimeError(f"openrouter {r.status_code}: {err}")
                    break
                except (httpx.TransportError, json.JSONDecodeError) as ex:
                    last_err = f"{type(ex).__name__}: {ex}"
                    await asyncio.sleep(min(60, 2 ** attempt))
            else:
                raise RuntimeError(f"openrouter gave up after retries: {last_err}")
        usd = float((data.get("usage") or {}).get("cost") or 0)
        db.execute("INSERT INTO spend VALUES (?, ?, ?)", (time.time(), model, usd))
        if _CACHE_PATH:
            db.execute("INSERT OR IGNORE INTO c VALUES (?, ?)", (key, json.dumps(data)))
        db.commit()
        return data, False
    finally:
        db.close()


async def _chat_openrouter(body: dict, x_caller: str | None):
    model = _or_model(body.get("model"))
    t0 = time.time()
    try:
        data, cached = await _openrouter(body, model)
    except Exception as e:  # noqa: BLE001
        _log({"ts": t0, "caller": x_caller, "model": model, "error": str(e)[:300]})
        return JSONResponse({"error": {"message": str(e), "type": "shim_error"}}, status_code=500)
    msg = dict(data["choices"][0].get("message") or {})
    rf = body.get("response_format")
    if rf and rf.get("type") in ("json_object", "json_schema") and isinstance(msg.get("content"), str):
        msg["content"] = _FENCE.sub("", msg["content"].strip())
    u = data.get("usage") or {}
    message = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        message["tool_calls"] = msg["tool_calls"]
    finish = data["choices"][0].get("finish_reason") or ("tool_calls" if msg.get("tool_calls") else "stop")
    resp = {"id": data.get("id") or f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion", "created": int(t0),
            "model": body.get("model") or model, "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": int(u.get("prompt_tokens") or 0), "completion_tokens": int(u.get("completion_tokens") or 0),
                      "total_tokens": int(u.get("total_tokens") or 0)}}
    usd = float(u.get("cost") or 0)
    _log({"ts": t0, "caller": x_caller, "model": model, "latency_s": round(time.time() - t0, 3),
          "prompt_tokens": resp["usage"]["prompt_tokens"], "completion_tokens": resp["usage"]["completion_tokens"],
          "provider_cached_tokens": int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0), "finish_reason": finish,
          "cost_usd": 0.0 if cached else usd, "cost_usd_uncached": usd, "cached": cached,
          "tool_calls": len(msg.get("tool_calls") or []), "json": bool(rf)})
    if body.get("stream"):
        async def gen():
            chunk = {"id": resp["id"], "object": "chat.completion.chunk", "created": resp["created"], "model": resp["model"],
                     "choices": [{"index": 0, "delta": message, "finish_reason": finish}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(resp)


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
    # deterministic ids (hash of the call, not uuid4): a tool-call id is echoed into the next prompt, so a random id would
    # make every later turn of a tool loop miss the prompt cache (round 3 lost the memory-tool replay to this)
    import hashlib
    return [{"id": "call_" + hashlib.sha256(f"{i}|{c.get('name')}|{json.dumps(c.get('arguments', {}), sort_keys=True)}".encode()).hexdigest()[:12],
             "type": "function", "function": {"name": c.get("name"), "arguments": json.dumps(c.get("arguments", {}))}} for i, c in enumerate(calls)]


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
    if _BACKEND == "openrouter":
        return await _chat_openrouter(body, x_caller)
    model = _model_alias(body.get("model"))
    rf = body.get("response_format")
    schema = rf.get("json_schema", {}).get("schema") if rf and rf.get("type") == "json_schema" else None
    system, prompt = _render(body.get("messages", []), body.get("tools"), rf, body.get("tool_choice"))
    t0 = time.time()
    try:
        data, cached = await _claude_cached(system, prompt, model, schema)
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
          "cost_usd": 0.0 if cached else data.get("total_cost_usd"), "cost_usd_uncached": data.get("total_cost_usd"), "cached": cached,
          "tool_calls": len(tool_calls or []),
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
