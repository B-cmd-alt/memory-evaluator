"""Tiny durable store: one JSON object per record, under a prefix. GCS in production, a local directory otherwise.

Write-once records (votes) and a read-modify-write ledger (daily budget) are all it needs. No database,
no migrations; the whole app's state is a bucket you can `ls`.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from .config import GCS_BUCKET, LOCAL_DATA


class Store:
    def __init__(self, bucket: str = GCS_BUCKET, local: Path = LOCAL_DATA):
        self.bucket_name = bucket
        self.local = Path(local)
        self._lock = threading.Lock()
        self._bucket = None
        if bucket:
            from google.cloud import storage
            self._bucket = storage.Client().bucket(bucket)
        else:
            self.local.mkdir(parents=True, exist_ok=True)

    # ── primitives ───────────────────────────────────────────────────────────────────────────
    def put(self, key: str, obj: dict) -> None:
        data = json.dumps(obj, sort_keys=True)
        if self._bucket:
            self._bucket.blob(key).upload_from_string(data, content_type="application/json")
        else:
            p = self.local / key
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(data)

    def get(self, key: str) -> dict | None:
        if self._bucket:
            b = self._bucket.blob(key)
            return json.loads(b.download_as_text()) if b.exists() else None
        p = self.local / key
        return json.loads(p.read_text()) if p.exists() else None

    def list(self, prefix: str) -> list[dict]:
        out = []
        if self._bucket:
            for b in self._bucket.list_blobs(prefix=prefix):
                try:
                    out.append(json.loads(b.download_as_text()))
                except ValueError:
                    pass
        else:
            d = self.local / prefix
            for p in sorted(d.glob("*.json")) if d.exists() else []:
                try:
                    out.append(json.loads(p.read_text()))
                except ValueError:
                    pass
        return out

    # ── helpers ──────────────────────────────────────────────────────────────────────────────
    def append(self, prefix: str, obj: dict) -> str:
        key = f"{prefix.rstrip('/')}/{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.json"
        self.put(key, obj)
        return key

    def update(self, key: str, fn) -> dict:
        """Read-modify-write under a process lock (good enough for a 1–2 instance service)."""
        with self._lock:
            cur = self.get(key) or {}
            new = fn(cur)
            self.put(key, new)
            return new


_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
