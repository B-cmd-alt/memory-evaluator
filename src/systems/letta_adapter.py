"""Letta (MemGPT lineage): OS-style self-managed memory blocks + archival memory.

STATUS: BLOCKED in this environment — `pip install letta` has no release compatible with the
container's Python 3.13, and Letta needs its own server process with an OpenAI-compatible model
endpoint. The adapter below is the intended integration (Letta Python client → local server →
shim); it raises a clear error until the environment can host it. Recorded honestly in the
feasibility report rather than faked.
"""
from __future__ import annotations

from src.systems.base_adapter import BaseAdapter


class LettaAdapter(BaseAdapter):
    name, family, year, kind = "letta", "OS-style self-managed memory", 2023, "modern"
    BLOCKED = "letta not installable on Python 3.13 in this container; needs a 3.11/3.12 env + `letta server`"

    def _on_reset(self):
        try:
            import letta  # noqa: F401
        except ImportError as e:
            raise RuntimeError(self.BLOCKED) from e
        raise RuntimeError("Letta client wiring not implemented yet: " + self.BLOCKED)

    async def query(self, question, timestamp=None):
        raise RuntimeError(self.BLOCKED)
