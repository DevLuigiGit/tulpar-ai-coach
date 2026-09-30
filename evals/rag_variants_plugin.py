"""Retrieval variants for evals/retrieval_eval.py, built through the production code (rag.variants → Index → retrieve).

    python evals/retrieval_eval.py run --plugin evals/rag_variants_plugin.py \
        --config v_default --config v_d512 --config v_hybrid ... --out evals/results/rag_matrix.json

Each config differs from production only by the settings in VARIANTS, so a winner can be switched on by settings.
"""

from __future__ import annotations

import sys

_host = sys.modules.get("__main__")
if not hasattr(_host, "register"):  # imported outside the evaluator's own process
    import retrieval_eval as _host  # type: ignore[no-redef]

register, RetrievalConfig = _host.register, _host.RetrievalConfig

VARIANTS: dict[str, tuple[dict, str]] = {
    "default": ({}, "production: Jina v3 1024-d dense, no rerank"),
    "d512": ({"rag_embed_dim": 512}, "Matryoshka 512-d (truncated + renormalised Jina vectors)"),
    "d256": ({"rag_embed_dim": 256}, "Matryoshka 256-d"),
    "hybrid": ({"rag_hybrid": True}, "dense + BM25 sparse in one Qdrant collection, RRF fusion"),
    "ctx": ({"rag_context_headers": True}, "context header prepended to the embedded chunk text"),
    "mq_en": ({"rag_multi_query": "en"}, "Russian query also searched in English, RRF fusion"),
    "late": ({"rag_late_chunking": True}, "Jina late chunking: chunks of a page window embedded together"),
    "hybrid_mq": ({"rag_hybrid": True, "rag_multi_query": "en"}, "hybrid + English multi-query"),
}


def _factory(name: str, overrides: dict, help: str):
    async def make(ctx, arg):
        from tulpar_ai.rag.variants import build_retriever, describe

        r = build_retriever(overrides=overrides, path=ctx.data_dir / "qdrant", rerank=False)
        points = await r.build()

        async def run(query: str, k: int) -> list[str]:
            return await r.retrieve_ids(query, k)

        meta = {"description": help, "overrides": overrides, "variant": describe(r.variant),
                "collection": r.collection, "points": points, "code_path": "tulpar_ai.rag.variants.Retriever"}
        return RetrievalConfig(f"v_{name}", run, meta, close=r.close)

    return make


for _name, (_ov, _help) in VARIANTS.items():
    register(f"v_{_name}", _help)(_factory(_name, _ov, _help))
