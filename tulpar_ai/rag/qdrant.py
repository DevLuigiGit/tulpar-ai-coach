"""One Qdrant client per storage location, shared by everything in the process.

Embedded mode (default) keeps vectors in AI_DATA_DIR/qdrant. Its file lock allows one client per directory,
so the RAG index and anything else that stores vectors (the answer cache) must share this instance.
QDRANT_URL switches to a Qdrant server — docker compose, a Railway service or Qdrant Cloud — with the same API;
nothing else in the code changes.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from ..config import get_settings

_clients: dict[str, QdrantClient] = {}


def location(path: Path | None = None) -> str:
    s = get_settings()
    return s.qdrant_url or str(path or Path(s.ai_data_dir) / "qdrant")


def get_client(path: Path | None = None) -> QdrantClient:
    s = get_settings()
    key = location(path)
    if key not in _clients:
        if s.qdrant_url:
            _clients[key] = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=30)
        else:
            _clients[key] = QdrantClient(path=key)
    return _clients[key]


def close_client(path: Path | None = None) -> None:
    client = _clients.pop(location(path), None)
    if client is not None:
        client.close()
