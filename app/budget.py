"""The visible, bounded cost of the live features.

A hard daily budget in USD for the whole site and a per-visitor daily request cap. When the budget is
spent the live endpoints refuse (the precomputed evidence keeps working), and the page shows the ledger.
"""
from __future__ import annotations

import datetime as dt
import threading

from .config import DAILY_BUDGET_USD, PER_IP_DAILY_LIVE
from .store import store


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


class Budget:
    def __init__(self):
        self._ip_counts: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def ledger(self) -> dict:
        d = store().get(f"budget/{_today()}.json") or {}
        return {"date": _today(), "spent_usd": round(float(d.get("usd", 0.0)), 4), "requests": int(d.get("requests", 0)),
                "daily_budget_usd": DAILY_BUDGET_USD, "remaining_usd": round(max(0.0, DAILY_BUDGET_USD - float(d.get("usd", 0.0))), 4)}

    def allow(self, ip: str) -> tuple[bool, str]:
        led = self.ledger()
        if led["remaining_usd"] <= 0:
            return False, f"today's live budget (${DAILY_BUDGET_USD:.2f}) is spent; precomputed evidence still works"
        with self._lock:
            n = self._ip_counts.get((ip, _today()), 0)
            if n >= PER_IP_DAILY_LIVE:
                return False, f"you have used today's {PER_IP_DAILY_LIVE} live requests from this address"
        return True, ""

    def charge(self, ip: str, usd: float) -> dict:
        with self._lock:
            k = (ip, _today())
            self._ip_counts[k] = self._ip_counts.get(k, 0) + 1
        return store().update(f"budget/{_today()}.json", lambda cur: {"usd": float(cur.get("usd", 0.0)) + float(usd),
                                                                       "requests": int(cur.get("requests", 0)) + 1})


_budget: Budget | None = None


def budget() -> Budget:
    global _budget
    if _budget is None:
        _budget = Budget()
    return _budget
