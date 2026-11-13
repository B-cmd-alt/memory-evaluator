"""The shim waits out a subscription usage-limit window and retries; any other error still fails at once."""
from __future__ import annotations

import asyncio

import pytest

from tools import openai_shim as shim

LIMIT = RuntimeError('claude exited 1: {"api_error":"usage_limit_reached","result":"You\'ve hit your 5-hour limit · resets 2am (UTC)"}')


def run(monkeypatch, outcomes, wait_s=3600):
    calls = []

    async def once(*a):
        calls.append(1)
        o = outcomes[len(calls) - 1]
        if isinstance(o, Exception):
            raise o
        return o

    async def no_sleep(_):
        return None

    monkeypatch.setattr(shim, "_claude_once", once)
    monkeypatch.setattr(shim.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(shim, "_log", lambda rec: None)
    monkeypatch.setattr(shim, "_LIMIT_WAIT_S", wait_s)
    return asyncio.run(shim._claude("s", "p", "sonnet", None)), calls


def test_usage_limit_is_waited_out(monkeypatch):
    out, calls = run(monkeypatch, [LIMIT, LIMIT, {"result": "ok"}])
    assert out == {"result": "ok"} and len(calls) == 3


def test_other_errors_fail_immediately(monkeypatch):
    with pytest.raises(RuntimeError, match="boom"):
        run(monkeypatch, [RuntimeError("claude exited 1: boom"), {"result": "never"}])


def test_gives_up_after_the_wait_budget(monkeypatch):
    with pytest.raises(RuntimeError, match="usage_limit_reached"):
        run(monkeypatch, [LIMIT] * 10, wait_s=shim._LIMIT_POLL_S * 2)
