"""One Qdrant client per storage location, shared by everything in the process.

Embedded mode (default) keeps vectors in AI_DATA_DIR/qdrant. Its file lock allows one client per directory,
so the RAG index and anything else that stores vectors (the answer cache) must share this instance.
QDRANT_URL switches to a Qdrant server — docker compose, a Railway service or Qdrant Cloud — with the same API;
nothing else in the code changes. If that server does not answer at startup, the process logs it and keeps
working on embedded Qdrant instead of escalating every question: the index is rebuilt locally.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from qdrant_client import QdrantClient

from ..config import get_settings

log = logging.getLogger(__name__)

_clients: dict[str, QdrantClient] = {}
_server: QdrantClient | None = None
_server_failed = False  # decided once per process: a flapping server must not split the index in two


def _server_client() -> QdrantClient | None:
    global _server, _server_failed
    s = get_settings()
    if not s.qdrant_url or _server_failed:
        return None
    if _server is None:
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=30)
        error: Exception | None = None
        for attempt in range(3):  # private DNS of a freshly started neighbour can lag a few seconds
            try:
                client.get_collections()
                _server = client
                return _server
            except Exception as e:  # noqa: BLE001 — any failure means «not usable right now»
                error = e
                time.sleep(2 * (attempt + 1))
        client.close()
        _server_failed = True
        log.warning("Qdrant server %s is unreachable (%s): using embedded Qdrant in AI_DATA_DIR", s.qdrant_url, error)
        return None
    return _server


def mode() -> str:
    return "server" if _server_client() else "embedded"


def location(path: Path | None = None) -> str:
    s = get_settings()
    return s.qdrant_url if _server_client() else str(path or Path(s.ai_data_dir) / "qdrant")


def get_client(path: Path | None = None) -> QdrantClient:
    server = _server_client()
    if server is not None:
        return server
    key = location(path)
    if key not in _clients:
        _clients[key] = QdrantClient(path=key)
    return _clients[key]


def close_client(path: Path | None = None) -> None:
    global _server
    if _server is not None:
        _server.close()
        _server = None
        return
    client = _clients.pop(location(path), None)
    if client is not None:
        client.close()
