"""Fill the on-disk embedding cache from an index that was already built, instead of paying Jina for it again.

    python tools/seed_emb_cache.py --qdrant data/final_nutrition/qdrant [--collection coach_jina3_400_p2_b064ec4f]

Only a default-variant Jina collection qualifies (`coach_jina3_<chunk>_p<N>_<fingerprint>`, no variant suffix): its
vectors are exactly `embed(chunk text, task=retrieval.passage)`, so each one is stored under the key the embedder
would compute for that text. Header-prefixed or late-chunked collections embed something else and are refused.
The source folder is copied first, so a running process holding the embedded Qdrant's lock is not disturbed.
Target: AI_DATA_DIR/emb_cache (RAG_EMBED_CACHE=passages|all then reads it).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = re.compile(r"^coach_jina3_\d+_p\d+_[0-9a-f]{8}$")


def seed(qdrant_dir: Path, collection: str | None = None) -> tuple[str, int, int]:
    from qdrant_client import QdrantClient

    from tulpar_ai.config import get_settings
    from tulpar_ai.rag.emb_cache import EmbeddingCache, cache_key

    s = get_settings()
    cache = EmbeddingCache(Path(s.ai_data_dir) / "emb_cache")
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "qdrant"
        shutil.copytree(qdrant_dir, copy)
        client = QdrantClient(path=str(copy))
        try:
            names = [c.name for c in client.get_collections().collections]
            if collection is None:
                candidates = sorted(n for n in names if BASE.match(n))
                if not candidates:
                    raise SystemExit(f"no default-variant Jina collection in {qdrant_dir}: {names}")
                collection = candidates[-1]
            if not BASE.match(collection):
                raise SystemExit(f"{collection} is not a default-variant Jina collection: its vectors are not plain passages")
            added = total = 0
            offset = None
            while True:
                points, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=True)
                for p in points:
                    total += 1
                    key = cache_key(s.embed_model, "retrieval.passage", p.payload["text"])
                    if cache.get(key) is None:
                        cache.put(key, list(p.vector))
                        added += 1
                if offset is None:
                    break
        finally:
            client.close()
    return collection, total, added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant", required=True, type=Path, help="folder of an embedded Qdrant (AI_DATA_DIR/qdrant)")
    ap.add_argument("--collection")
    args = ap.parse_args()
    name, total, added = seed(args.qdrant, args.collection)
    print(f"{name}: {total} vectors read, {added} added to the embedding cache")


if __name__ == "__main__":
    main()
