"""QDRANT_URL that does not answer must not take RAG down: the process falls back to embedded Qdrant."""

from __future__ import annotations


def test_unreachable_server_falls_back_to_embedded(tmp_path, monkeypatch):
    from tulpar_ai.config import get_settings
    from tulpar_ai.rag import qdrant

    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")  # discard port: nothing listens there
    monkeypatch.setenv("AI_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr(qdrant.time, "sleep", lambda s: None)
    monkeypatch.setattr(qdrant, "_server", None)
    monkeypatch.setattr(qdrant, "_server_failed", False)
    try:
        client = qdrant.get_client(tmp_path / "qdrant")
        assert qdrant.mode() == "embedded"
        assert qdrant.location(tmp_path / "qdrant") == str(tmp_path / "qdrant")
        assert client.get_collections().collections == []
    finally:
        qdrant.close_client(tmp_path / "qdrant")
        get_settings.cache_clear()
