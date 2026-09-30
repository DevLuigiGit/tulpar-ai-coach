"""Embedding map for the defense slides and docs/rag: where Russian client questions land in the Jina vector space
relative to the corpus chunks (exercise cards, Tulpar nutrition rules, the English WHO 2020 PDF).

    python evals/embedding_map.py --copy-from data/final_nutrition/qdrant   # first run: copy an existing index
    python evals/embedding_map.py                                          # rerun: everything from disk, no network
    python evals/embedding_map.py --method pca                             # without umap-learn

No corpus re-embedding (the Jina quota is shared): corpus vectors are READ from a copy of an existing embedded Qdrant
index (AI_DATA_DIR/qdrant, the copy has no .lock, so a running app is not disturbed). Only the example questions are
embedded, with the same model and task as production (jina-embeddings-v3, retrieval.query), and every vector is cached
in AI_DATA_DIR/emb_cache keyed by model + task + options + text hash, so a rerun makes zero API calls.

Retrieval here is the production first pass for Jina: dense cosine top-k, reranker off (RAG_RERANK=auto), no query
rewrite. The numpy ranking is cross-checked against Qdrant's own query_points for every question.
A hit uses the qa eval's rule (evals/run.py _hit_rank: WHO page ±1, exercise title exact).

Outputs (docs/rag/): embedding_map.png/.svg (2-D map), cosine_bars.png/.svg (cosine: question ↔ its WHO chunk vs
random chunks), embedding_map.json (every number shown in the pictures). Plotting needs requirements-viz.txt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLDEN = ROOT / "evals" / "golden" / "qa.jsonl"
OUT = ROOT / "docs" / "rag"

# 8 cross-lingual questions to the English WHO PDF (the weakest area) + 2 technique + 2 nutrition, all from the golden set.
QUERY_IDS = ("q40", "q42", "q43", "q44", "q45", "q47", "q49", "q51", "q01", "q18", "q34", "q39")
# Short slide labels (paraphrases); the exact golden questions are in embedding_map.json and docs/rag/README.md.
SHORT = {
    "q40": "Сколько минут в неделю нужно взрослому",
    "q42": "Сколько раз в неделю силовые",
    "q43": "Маме 68 лет: что кроме ходьбы",
    "q44": "Ребёнок 10 лет: сколько двигаться",
    "q45": "Сидячая работа: что говорит ВОЗ",
    "q47": "Беременность: сколько заниматься",
    "q49": "Диабет 2 типа: сколько активности",
    "q51": "Умеренная или интенсивная нагрузка",
    "q01": "Жим лёжа: угол локтей",
    "q18": "Больные колени и присед со штангой",
    "q34": "Набор массы: сколько в неделю",
    "q39": "Как считается норма калорий",
}
EMBEDDER_ID, EMBED_MODEL, QUERY_TASK, EMBED_OPTS = "jina3", "jina-embeddings-v3", "retrieval.query", "api-default-1024"

# Slide palette: brand surface and accent; the three source hues pass the dataviz validator all-pairs on #141416.
BG, ACCENT, INSET_BG = "#141416", "#FF5000", "#1C1C20"
INK, INK2, MUTED, RULE, GREY_BAR = "#F4F4F5", "#C9C9D1", "#8E8E99", "#2C2C31", "#5D5D66"
SOURCES = {  # payload source → (legend label, colour, marker)
    "who2020": ("Руководство ВОЗ 2020 — английский PDF", "#3987e5", "o"),
    "exercises": ("Карточки упражнений — русский", "#199e70", "o"),
    "nutrition": ("Правила питания Tulpar — русский", "#c98500", "D"),
}
SOURCE_TEXT = {"who2020": "PDF ВОЗ, английский", "exercises": "Карточки упражнений,\nрусский",
               "nutrition": "Правила питания,\nрусский"}  # direct labels on the map
FONTS = ["PT Sans", "Arial", "DejaVu Sans"]


# ── index and cache ──────────────────────────────────────────────────────────
def expected_collection(embedder_id: str = EMBEDDER_ID, pdf_chunk: int = 400) -> str:
    """The collection the app itself would open for the current corpus (see tulpar_ai/rag/index.py)."""
    from parsing.chunker import PARSING_VERSION
    from tulpar_ai.rag.index import corpus_fingerprint

    return f"coach_{embedder_id}_{pdf_chunk}_p{PARSING_VERSION}_{corpus_fingerprint()}"


def copy_index(src: Path, dst: Path) -> None:
    """Private copy of an embedded Qdrant folder, without the lock file of the process that owns the original."""
    if not (src / "meta.json").exists():
        raise SystemExit(f"{src} is not an embedded Qdrant folder (no meta.json)")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".lock"))


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def load_corpus(client, collection: str) -> tuple[list[dict], np.ndarray]:
    """All points of the collection: payloads and L2-normalised vectors, in a stable order (by chunk id)."""
    names = [c.name for c in client.get_collections().collections]
    if collection not in names:
        raise SystemExit(f"collection {collection} is not in the index copy (found: {', '.join(names) or 'none'}); "
                         "copy a fresh index with --copy-from or pass --collection")
    points, offset = [], None
    while True:
        batch, offset = client.scroll(collection, limit=512, offset=offset, with_payload=True, with_vectors=True)
        points.extend(batch)
        if offset is None:
            break
    points.sort(key=lambda p: str(p.payload.get("id")))
    return [dict(p.payload) for p in points], normalize(np.asarray([p.vector for p in points], dtype=np.float32))


class VectorCache:
    """One .npy per (model, task, options, text): a vector paid for once is never requested again."""

    def __init__(self, root: Path):
        self.root = root

    def path(self, text: str, model: str, task: str, opts: str) -> Path:
        key = hashlib.sha256("\x1f".join((model, task, opts, text)).encode()).hexdigest()[:32]
        return self.root / f"{model}__{task}__{opts}__{key}.npy"

    def get(self, text: str, model: str, task: str, opts: str) -> np.ndarray | None:
        p = self.path(text, model, task, opts)
        return np.load(p) if p.exists() else None

    def put(self, text: str, model: str, task: str, opts: str, vec) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.path(text, model, task, opts), np.asarray(vec, dtype=np.float32))


EmbedFn = Callable[[list[str], str], Awaitable[list[list[float]]]]


async def embed_with_cache(texts: list[str], *, cache: VectorCache, embed: EmbedFn | None, model: str = EMBED_MODEL,
                           task: str = QUERY_TASK, opts: str = EMBED_OPTS) -> tuple[np.ndarray, int]:
    """Vectors for texts; only cache misses go to `embed` (one batch). Returns (vectors, texts sent to the API)."""
    missing = [t for t in dict.fromkeys(texts) if cache.get(t, model, task, opts) is None]
    if missing:
        if embed is None:
            raise SystemExit(f"{len(missing)} question(s) are not cached and JINA_API_KEY is not set")
        for t, v in zip(missing, await embed(missing, task), strict=True):
            cache.put(t, model, task, opts, v)
    return np.stack([cache.get(t, model, task, opts) for t in texts]), len(missing)


# ── retrieval analysis ───────────────────────────────────────────────────────
def _hit_rank_fn():
    """The qa eval's hit rule, imported from evals/run.py so the two can never disagree."""
    spec = importlib.util.spec_from_file_location("evals_run_for_map", ROOT / "evals" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._hit_rank


def _brief(p: dict, score: float) -> dict:
    return {"id": p.get("id"), "source": p.get("source"), "title": p.get("title"), "page": p.get("page"),
            "score": round(float(score), 4)}


def analyze_question(item: dict, qvec: np.ndarray, corpus: np.ndarray, payloads: list[dict], *, hit_rank,
                     rng: np.random.Generator, n_random: int = 200, top_n: int = 4) -> dict:
    """Dense ranking of one question over the whole corpus + the relevant chunk + a random-chunk baseline."""
    scores = corpus @ normalize(qvec)
    order = np.argsort(-scores, kind="stable")
    ranked = [payloads[i] for i in order]
    row = {"id": item["id"], "question": item["question"], "short": SHORT.get(item["id"], item["question"]),
           "tags": item.get("tags", []), "expected_sources": item.get("expected_sources", []),
           "top": [_brief(payloads[i], scores[i]) for i in order[:top_n]]}
    expected = item.get("expected_sources") or []
    rank = hit_rank(ranked, expected) if expected else None
    row["hit_rank"] = rank
    row[f"hit@{top_n}"] = rank is not None and rank <= top_n
    row["hit@3"] = rank is not None and rank <= 3
    row["relevant"] = _brief(ranked[rank - 1], scores[order[rank - 1]]) if rank else None
    relevant = {i for i in range(len(payloads)) if expected and hit_rank([payloads[i]], expected) == 1}
    pool = np.array([i for i in range(len(payloads)) if i not in relevant])
    sample = rng.choice(pool, size=min(n_random, len(pool)), replace=False)
    row["random"] = _stats(scores[sample])
    who_pool = np.array([i for i in pool if payloads[i].get("source") == "who2020"])
    if len(who_pool):
        row["random_who"] = _stats(scores[rng.choice(who_pool, size=min(n_random, len(who_pool)), replace=False)])
    return row


def _stats(x: np.ndarray) -> dict:
    return {"n": int(len(x)), "mean": round(float(np.mean(x)), 4), "p5": round(float(np.percentile(x, 5)), 4),
            "p95": round(float(np.percentile(x, 95)), 4), "max": round(float(np.max(x)), 4)}


def qdrant_top_ids(client, collection: str, qvec: np.ndarray, limit: int) -> list[str]:
    res = client.query_points(collection, query=normalize(qvec).tolist(), limit=limit, with_payload=True)
    return [p.payload.get("id") for p in res.points]


def summarize(rows: list[dict], top_n: int = 4) -> dict:
    who = [r for r in rows if "who" in r["tags"]]
    out = {"questions": len(rows), "who_questions": len(who)}
    if who:
        rel = [r["relevant"]["score"] for r in who if r["relevant"]]
        out.update({
            "who_hit@3": sum(r["hit@3"] for r in who), f"who_hit@{top_n}": sum(r[f"hit@{top_n}"] for r in who),
            "who_top3_chunks_from_who_pdf": sum(t["source"] == "who2020" for r in who for t in r["top"][:3]),
            "who_top3_chunks_total": 3 * len(who),
            "who_cos_relevant_median": round(float(np.median(rel)), 4) if rel else None,
            "who_cos_random_mean_median": round(float(np.median([r["random"]["mean"] for r in who])), 4),
            "who_cos_random_p95_max": round(float(max(r["random"]["p95"] for r in who)), 4),
        })
    out["hit@3_all"] = sum(r["hit@3"] for r in rows)
    out["qdrant_top4_match"] = sum(bool(r.get("qdrant_top4_match")) for r in rows)
    return out


# ── 2-D projection ───────────────────────────────────────────────────────────
def pca_2d(corpus: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = corpus.mean(axis=0)
    _, _, vt = np.linalg.svd(corpus - mean, full_matrices=False)
    return (corpus - mean) @ vt[:2].T, (extra - mean) @ vt[:2].T


def project(corpus: np.ndarray, queries: np.ndarray, method: str = "umap", seed: int = 42,
            n_neighbors: int = 15, min_dist: float = 0.1, fit: str = "joint") -> tuple[np.ndarray, np.ndarray, dict]:
    """2-D coordinates for the chunks and the questions.

    fit="joint" (default): UMAP is fitted on chunks + questions together. On this index that keeps more of each
    question's true top-3 next to it on the map than fit="transform" (fit on the chunks, place the questions with
    UMAP.transform) — compare with neighbour_preservation, numbers in docs/rag/README.md.
    PCA (no extra dependency) is fitted on the chunks only; the questions use the same linear projection."""
    if method == "umap":
        try:
            import umap
        except ImportError:
            print("umap-learn is not installed: falling back to PCA", file=sys.stderr)
            method = "pca"
    if method == "umap":
        import warnings

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=seed)
        q = normalize(queries)
        with warnings.catch_warnings():  # «n_jobs overridden by random_state»: the fixed seed is the point
            warnings.simplefilter("ignore", UserWarning)
            if fit == "transform":
                c2, q2 = reducer.fit_transform(corpus), reducer.transform(q)
            else:
                xy = reducer.fit_transform(np.vstack([corpus, q]))
                c2, q2 = xy[:len(corpus)], xy[len(corpus):]
        return c2, q2, {"method": "umap", "n_neighbors": n_neighbors, "min_dist": min_dist, "metric": "cosine",
                        "seed": seed, "fit": "chunks + questions" if fit == "joint" else "chunks; questions via transform"}
    c2, q2 = pca_2d(corpus, normalize(queries))
    return c2, q2, {"method": "pca", "fit_on": "chunks", "questions": "same linear projection"}


def neighbour_preservation(corpus: np.ndarray, queries: np.ndarray, c2: np.ndarray, q2: np.ndarray,
                           k: int = 15, q_top: int = 3, q_window: int = 10) -> dict:
    """How much the 2-D picture lies, in numbers.

    chunk_knn_recall: share of each chunk's k nearest neighbours (cosine, 1024-d) that are also among its k nearest
    points on the map. question_top3_in_map: share of each question's true top-3 chunks that are among its
    q_window nearest chunks on the map."""
    sims = corpus @ corpus.T
    np.fill_diagonal(sims, -np.inf)
    true_nn = np.argsort(-sims, axis=1)[:, :k]
    d2 = ((c2[:, None, :] - c2[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    map_nn = np.argsort(d2, axis=1)[:, :k]
    chunk = np.mean([len(set(a) & set(b)) / k for a, b in zip(true_nn, map_nn)])
    top = np.argsort(-(corpus @ normalize(queries).T), axis=0)[:q_top].T
    near = np.argsort(((q2[:, None, :] - c2[None, :, :]) ** 2).sum(-1), axis=1)[:, :q_window]
    question = np.mean([len(set(a) & set(b)) / q_top for a, b in zip(top, near)])
    return {f"chunk_knn_recall@{k}": round(float(chunk), 3),
            f"question_top{q_top}_within_{q_window}_nearest_on_map": round(float(question), 3)}




def compare_with_eval(rows: list[dict], results: Path, top_n: int = 4) -> dict | None:
    """Cross-check with a saved qa run: evals/run.py stores the hit rank within the first top_n (None beyond)."""
    if not results.exists():
        return None
    saved = {}
    for r in json.loads(results.read_text(encoding="utf-8")).get("rows", []):
        saved.setdefault(r["id"], r.get("rank"))
    same = [r["id"] for r in rows if r["id"] in saved
            and saved[r["id"]] == (r["hit_rank"] if r["hit_rank"] and r["hit_rank"] <= top_n else None)]
    return {"file": str(results.relative_to(ROOT)) if results.is_relative_to(ROOT) else results.name,
            "compared": sum(r["id"] in saved for r in rows), "same_rank": len(same)}


# ── pictures ─────────────────────────────────────────────────────────────────
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": FONTS, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                         "text.color": INK, "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "svg.fonttype": "path", "svg.hashsalt": "tulpar-embedding-map"})
    return plt


def _src_label(t: dict, width: int = 16) -> str:
    if t["source"] == "who2020":
        return f"ВОЗ с.{t['page']}"
    if t["source"] == "nutrition":
        return textwrap.shorten("Питание: " + str(t["title"]).split("—")[-1].strip(), width, placeholder="…")
    return textwrap.shorten(str(t["title"]), width, placeholder="…")




Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in display pixels


def _overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _grow(box: Box, by: float) -> Box:
    return (box[0] - by, box[1] - by, box[2] + by, box[3] + by)


def _within(points: np.ndarray, box: Box) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return ((points[:, 0] >= box[0]) & (points[:, 0] <= box[2]) & (points[:, 1] >= box[1])
            & (points[:, 1] <= box[3]))


def _inside(points: np.ndarray, box: Box) -> int:
    return int(_within(points, box).sum())


def _fits(box: Box, frame: Box) -> bool:
    return frame[0] <= box[0] and box[2] <= frame[2] and frame[1] <= box[1] and box[3] <= frame[3]


def place_labels(stars: np.ndarray, label_w: list[float], label_h: float, star_r: float,
                 obstacles: list[Box] = (), frame: Box | None = None) -> list[tuple[float, float]]:
    """Greedy label placement in display pixels: each number goes to the nearest candidate spot that overlaps the
    fewest other numbers, stars and obstacles, stays inside the frame and is not closer to another star than to
    its own (a number between two stars reads as the wrong one). Returns (dx, dy) offsets."""
    dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1.35, 0), (-1.35, 0), (0, 1.35), (0, -1.35)]
    cands = [(dx * star_r * k, dy * star_r * k) for k in (1.0, 1.9, 2.8, 3.8) for dx, dy in dirs]
    star_boxes = [(x - star_r, y - star_r, x + star_r, y + star_r) for x, y in stars]
    crowd = [int((np.hypot(*(stars - s).T) < 4 * star_r).sum()) for s in stars]
    placed: list[Box] = []
    offsets: list[tuple[float, float]] = [(0.0, 0.0)] * len(stars)
    for i in sorted(range(len(stars)), key=lambda i: -crowd[i]):
        best = None
        for rank, (dx, dy) in enumerate(cands):
            cx, cy = stars[i][0] + dx, stars[i][1] + dy
            box = (cx - label_w[i] / 2, cy - label_h / 2, cx + label_w[i] / 2, cy + label_h / 2)
            cost = sum(_overlap(box, b) for b in [*placed, *star_boxes, *obstacles])
            if frame is not None and not _fits(box, frame):
                cost += 1e6
            dist = np.hypot(stars[:, 0] - cx, stars[:, 1] - cy)
            ambiguous = bool(len(stars) > 1 and np.delete(dist, i).min() < dist[i] * 1.15)
            key = (cost > 0, ambiguous, cost, rank)
            if best is None or key < best[0]:
                best = (key, (dx, dy), box)
        offsets[i] = best[1]
        placed.append(best[2])
    return offsets


def crowded_group(stars: np.ndarray, radius: float) -> list[int]:
    """The largest set of stars chained by distance < radius (display pixels): these get a zoomed inset."""
    parent = list(range(len(stars)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(stars)):
        for j in range(i + 1, len(stars)):
            if np.hypot(*(stars[i] - stars[j])) < radius:
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(stars)):
        groups.setdefault(root(i), []).append(i)
    return max(groups.values(), key=len)


def empty_spot(points: np.ndarray, frame: Box, w: float, h: float, avoid: Box, step: float = 16.0,
               margin: float = 12.0) -> Box:
    """Where a w×h box (display pixels) covers the fewest marks and not the `avoid` box; on ties, the spot with the
    most clearance around it, so the inset reads as a separate picture and not as more of the map."""
    best = None
    for x0 in np.arange(frame[0] + margin, frame[2] - w - margin + 1, step):
        for y0 in np.arange(frame[1] + margin, frame[3] - h - margin + 1, step):
            box = (x0, y0, x0 + w, y0 + h)
            cost = _inside(points, box) + (1e6 if _overlap(box, avoid) > 0 else 0)
            dx = np.maximum(np.maximum(box[0] - points[:, 0], points[:, 0] - box[2]), 0)
            dy = np.maximum(np.maximum(box[1] - points[:, 1], points[:, 1] - box[3]), 0)
            clearance = float(np.hypot(dx, dy).min()) if len(points) else 0.0
            if best is None or (cost, -clearance) < best[0]:
                best = ((cost, -clearance), box)
    return best[1]


def _text_box(x: float, y: float, text: str, size_pt: float, s: float, ha: str, va: str) -> Box:
    lines = text.split("\n")
    w = max(len(line) for line in lines) * size_pt * 0.53 * s
    h = len(lines) * size_pt * 1.25 * s
    x0 = {"left": x, "center": x - w / 2, "right": x - w}[ha]
    y0 = {"bottom": y, "center": y - h / 2, "top": y - h}[va]
    return (x0, y0, x0 + w, y0 + h)


def plot_map(c2: np.ndarray, q2: np.ndarray, payloads: list[dict], rows: list[dict], meta: dict, out: Path) -> list[Path]:
    plt = _style()
    from matplotlib import patheffects as pe
    from matplotlib.lines import Line2D

    dpi = 120
    s = dpi / 72  # display pixels per point
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    ax = fig.add_axes((0.025, 0.05, 0.60, 0.78))
    panel = fig.add_axes((0.655, 0.05, 0.335, 0.78))
    panel.axis("off")
    src = np.array([p.get("source") for p in payloads])
    counts = {k: int((src == k).sum()) for k in SOURCES}
    index_of = {p.get("id"): i for i, p in enumerate(payloads)}
    halo = [pe.withStroke(linewidth=5, foreground=BG)]
    star_r = 11 * s

    def draw(a):
        a.set_xticks([]), a.set_yticks([])
        for side in a.spines.values():
            side.set_color(RULE)
        for key, size, alpha, z in (("who2020", 10, 0.6, 2), ("exercises", 14, 0.9, 2), ("nutrition", 90, 1.0, 3)):
            m = src == key
            a.scatter(c2[m, 0], c2[m, 1], s=size, c=SOURCES[key][1], marker=SOURCES[key][2], alpha=alpha, zorder=z,
                      linewidths=1.0 if key == "nutrition" else 0, edgecolors=BG)
        for r, (qx, qy) in zip(rows, q2):
            for t in r["top"][:3]:
                j = index_of[t["id"]]
                a.plot([qx, c2[j, 0]], [qy, c2[j, 1]], color=ACCENT, lw=1.4, alpha=0.7, zorder=4, solid_capstyle="round")
                a.scatter([c2[j, 0]], [c2[j, 1]], s=48 if t["source"] != "nutrition" else 90, c=SOURCES[t["source"]][1],
                          marker=SOURCES[t["source"]][2], edgecolors=INK, linewidths=1.3, zorder=5)
        a.scatter(q2[:, 0], q2[:, 1], s=520, marker="*", c=ACCENT, edgecolors=INK, linewidths=1.2, zorder=6)

    def numbers(a, idx, obstacles, frame):
        px = a.transData.transform(q2[idx])
        widths = [(9.5 * len(str(i + 1)) + 4) * s for i in idx]
        offsets = place_labels(px, widths, 17 * s, star_r, obstacles, frame)
        for i, (dx, dy) in zip(idx, offsets):
            far = np.hypot(dx, dy) > 20 * s
            a.annotate(str(i + 1), q2[i], xytext=(dx / s, dy / s), textcoords="offset points", ha="center",
                       va="center", fontsize=15, fontweight="bold", color=INK, zorder=8, path_effects=halo,
                       annotation_clip=False,  # a star cut by the inset edge keeps its number
                       arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2, shrinkA=6, shrinkB=10) if far else None)

    draw(ax)
    allx, ally = np.r_[c2[:, 0], q2[:, 0]], np.r_[c2[:, 1], q2[:, 1]]
    pad_x, pad_y = np.ptp(allx) * 0.05, np.ptp(ally) * 0.06
    ax.set_xlim(allx.min() - pad_x, allx.max() + pad_x)
    ax.set_ylim(ally.min() - pad_y, ally.max() + pad_y)
    frame = tuple(ax.bbox.extents)
    pts, stars = ax.transData.transform(c2), ax.transData.transform(q2)

    def source_labels(a, keys, obstacles: list[Box], frame: Box) -> list[Box]:
        """Direct labels beside the bulk of each cloud (10–90th percentile box), where they cover the fewest marks."""
        apts, astars, boxes = a.transData.transform(c2), a.transData.transform(q2), []
        for key in keys:
            p = apts[(src == key) & _within(apts, frame)]
            if not len(p):
                continue
            text, gap = SOURCE_TEXT[key], 12 * s
            lo, hi, mid = np.percentile(p, 10, axis=0), np.percentile(p, 90, axis=0), np.median(p, axis=0)
            best = None
            for x, y, ha, va in ((mid[0], hi[1] + gap, "center", "bottom"), (mid[0], lo[1] - gap, "center", "top"),
                                 (lo[0] - gap, mid[1], "right", "center"), (hi[0] + gap, mid[1], "left", "center")):
                box = _text_box(x, y, text, 13, s, ha, va)
                cost = (_inside(apts, box) + 100 * _inside(astars, _grow(box, 2 * star_r))
                        + sum(_overlap(box, b) for b in obstacles + boxes))
                key_ = (not _fits(box, frame), cost)
                if best is None or key_ < best[0]:
                    best = (key_, (x, y, ha, va), box)
            (x, y, ha, va), box = best[1], best[2]
            a.annotate(text, a.transData.inverted().transform((x, y)), ha=ha, va=va, fontsize=13, color=INK2,
                       zorder=7, path_effects=halo, linespacing=1.05)
            boxes.append(box)
        return boxes

    # Stars that sit on top of each other get a zoomed inset in the emptiest part of the map.
    group = crowded_group(stars, 2.6 * star_r)
    if len(group) < 3:
        group = []
    obstacles: list[Box] = []
    in_inset: list[str] = []
    if group:
        g = stars[group]
        gx0, gy0 = g.min(axis=0) - 30 * s
        gx1, gy1 = g.max(axis=0) + 30 * s
        iw, ih = 0.48 * ax.bbox.width, 0.50 * ax.bbox.height
        zoom = min(iw / (gx1 - gx0), ih / (gy1 - gy0))
        cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
        src_box = (cx - iw / zoom / 2, cy - ih / zoom / 2, cx + iw / zoom / 2, cy + ih / zoom / 2)
        spot = empty_spot(np.vstack([pts, stars]), frame, iw, ih, src_box)
        (ax0, ay0), (ax1, ay1) = ax.transAxes.inverted().transform([spot[:2], spot[2:]])
        ins = ax.inset_axes((ax0, ay0, ax1 - ax0, ay1 - ay0))
        ins.set_facecolor(INSET_BG)
        draw(ins)
        (dx0, dy0), (dx1, dy1) = ax.transData.inverted().transform([src_box[:2], src_box[2:]])
        ins.set_xlim(dx0, dx1)
        ins.set_ylim(dy0, dy1)
        for side in ins.spines.values():
            side.set_color(INK2)
        for c in ax.indicate_inset_zoom(ins, edgecolor=INK2, alpha=0.9, linewidth=1.0).connectors:
            c.set_linewidth(0.8)
        ins.text(0.02, 0.975, f"увеличено ×{zoom:.1f}", transform=ins.transAxes, fontsize=12, color=INK2, va="top",
                 zorder=9)
        # A source whose points mostly sit in the zoomed square is labelled inside the inset.
        in_inset = [k for k in SOURCES if (src == k).any() and _within(pts[src == k], src_box).mean() > 0.5]
        iframe = tuple(ins.bbox.extents)
        title_box = (iframe[0], iframe[3] - 24 * s, iframe[0] + 110 * s, iframe[3])
        inset_boxes = source_labels(ins, in_inset, [title_box], iframe)
        seen = _grow(src_box, star_r / zoom)  # a star this close outside the square still shows at the inset edge
        numbers(ins, list(np.flatnonzero(_within(stars, seen))), [title_box, *inset_boxes], iframe)
        obstacles += [spot, src_box]

    obstacles += source_labels(ax, [k for k in SOURCES if k not in in_inset], obstacles, frame)
    numbers(ax, [i for i in range(len(q2)) if i not in group], obstacles, frame)

    fig.text(0.025, 0.955, "Русские вопросы находят английские страницы ВОЗ", fontsize=27, fontweight="bold", color=INK,
             va="top")
    how = "UMAP" if meta["method"] == "umap" else "PCA"
    fig.text(0.025, 0.895, f"{sum(counts.values())} фрагментов корпуса и {len(rows)} вопросов клиентов · Jina "
             f"jina-embeddings-v3, 1024 измерения → {how} в 2-D · линии ведут к 3 ближайшим фрагментам по косинусу "
             "в 1024 измерениях", fontsize=13.5, color=INK2, va="top")
    handles = [Line2D([], [], ls="", marker=SOURCES[k][2], ms=8, mfc=SOURCES[k][1], mec=SOURCES[k][1],
                      label=f"{SOURCES[k][0]} · {counts[k]}") for k in SOURCES]
    handles.append(Line2D([], [], ls="", marker="*", ms=17, mfc=ACCENT, mec=INK, mew=1.0,
                          label="вопрос клиента, русский · линии → топ-3"))
    panel.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.01, 1.0), frameon=False, fontsize=12.5,
                 labelcolor=INK2, handletextpad=0.4, borderaxespad=0, labelspacing=0.4)
    y, step = 0.775, 0.0655
    for n, r in enumerate(rows, 1):
        panel.text(0.0, y, f"{n}", fontsize=14.5, fontweight="bold", color=ACCENT, va="top")
        panel.text(0.065, y, r["short"], fontsize=14.5, color=INK, va="top")
        tops = "  ·  ".join(f"{_src_label(t)} {t['score']:.2f}" for t in r["top"][:3])
        panel.text(0.065, y - 0.03, tops, fontsize=11.5, color=MUTED, va="top")
        y -= step
    fig.text(0.025, 0.012, "Оси без единиц: 2-D-проекция искажает расстояния. Мерило — косинусы в 1024 измерениях "
             "(в списке справа и на рис. cosine_bars).", fontsize=11, color=MUTED, va="bottom")
    return _save(fig, out / "embedding_map", plt)


def plot_bars(rows: list[dict], min_score: float, n_random: int, seed: int, top_n: int, out: Path) -> list[Path]:
    plt = _style()
    from matplotlib.patches import Patch

    who = [r for r in rows if "who" in r["tags"] and r["relevant"]]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes((0.29, 0.12, 0.69, 0.66))
    ys = np.arange(len(who))[::-1].astype(float)
    h, gap = 0.34, 0.02
    rel = np.array([r["relevant"]["score"] for r in who])
    rnd = np.array([r["random"]["mean"] for r in who])
    lo = np.array([r["random"]["p5"] for r in who])
    hi = np.array([r["random"]["p95"] for r in who])
    y_rel, y_rnd = ys + h / 2 + gap, ys - h / 2 - gap
    ax.barh(y_rel, rel, height=h, color=ACCENT, zorder=3)
    ax.barh(y_rnd, rnd, height=h, color=GREY_BAR, zorder=3)
    ax.hlines(y_rnd, lo, hi, color=INK2, lw=1.5, zorder=4)
    ax.vlines(np.r_[lo, hi], np.r_[y_rnd, y_rnd] - 0.07, np.r_[y_rnd, y_rnd] + 0.07, color=INK2, lw=1.5, zorder=4)
    for yy, v, r in zip(y_rel, rel, who):
        ax.text(v + 0.008, yy, f"{v:.2f}", va="center", fontsize=14.5, color=INK, fontweight="bold")
        where = f"место {r['hit_rank']} в выдаче" + ("" if r["hit_rank"] <= top_n else f" — вне топ-{top_n}")
        ax.text(v + 0.068, yy, f"ВОЗ с.{r['relevant']['page']} · {where}", va="center", fontsize=12, color=MUTED)
    for yy, v, b in zip(y_rnd, rnd, hi):
        ax.text(b + 0.008, yy, f"{v:.2f}", va="center", fontsize=12.5, color=INK2)
    ax.axvline(min_score, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(min_score + 0.006, len(who) - 0.42, f"rag_min_score = {min_score:g}: ниже — вопрос переписывается",
            fontsize=11.5, color=INK2, va="bottom")
    ax.set_yticks(ys, [r["short"] for r in who], fontsize=14.5, color=INK)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(min(0.0, float(lo.min()) - 0.02), 1.0)
    ax.set_ylim(-0.65, len(who) - 0.25)
    ax.set_xlabel("косинусное сходство, Jina v3, 1024 измерения", fontsize=13, color=INK2)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="x", color=RULE, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    fig.text(0.03, 0.955, "Русский вопрос ближе к нужной английской странице, чем к случайной", fontsize=27,
             fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.895, f"{len(who)} вопросов из evals/golden/qa.jsonl к PDF ВОЗ · вопрос — retrieval.query, "
             "фрагменты — retrieval.passage · плотный поиск без реранкера, как в проде", fontsize=13.5, color=INK2,
             va="top")
    ax.legend(handles=[Patch(color=ACCENT, label="нужный фрагмент ВОЗ: лучший с эталонной страницы (±1)"),
                       Patch(color=GREY_BAR, label=f"{n_random} случайных фрагментов: среднее, усы 5–95-й перцентиль")],
              loc="lower right", bbox_to_anchor=(1.0, 1.01), frameon=False, fontsize=12.5, labelcolor=INK2)
    fig.text(0.03, 0.015, f"Случайные фрагменты — из всего корпуса без эталонных страниц вопроса, seed {seed}.",
             fontsize=11, color=MUTED, va="bottom")
    return _save(fig, out / "cosine_bars", plt)


def _save(fig, stem: Path, plt) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=120, metadata={"Software": None})
    fig.savefig(paths[1], metadata={"Date": None, "Creator": None})
    plt.close(fig)
    return paths


# ── entry point ──────────────────────────────────────────────────────────────
def load_golden(path: Path, ids: tuple[str, ...]) -> list[dict]:
    items = {d["id"]: d for d in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}
    missing = [i for i in ids if i not in items]
    if missing:
        raise SystemExit(f"not in {path.name}: {', '.join(missing)}")
    return [items[i] for i in ids]


async def run(args) -> dict:
    from qdrant_client import QdrantClient

    from tulpar_ai.config import get_settings

    data_dir = Path(os.environ["AI_DATA_DIR"])
    qdir = data_dir / "qdrant"
    if args.copy_from:
        copy_index(Path(args.copy_from), qdir)
    if not (qdir / "meta.json").exists():
        raise SystemExit(f"no index copy in {qdir}: run once with --copy-from data/final_nutrition/qdrant")
    s = get_settings()
    if s.embed_model != EMBED_MODEL:
        raise SystemExit(f"EMBED_MODEL={s.embed_model}, but the {EMBEDDER_ID} index was built with {EMBED_MODEL}")
    collection = args.collection or expected_collection()
    client = QdrantClient(path=str(qdir))
    try:
        payloads, corpus = load_corpus(client, collection)
        golden = load_golden(Path(args.golden), tuple(args.ids.split(",")) if args.ids else QUERY_IDS)
        embed = None
        if s.jina_api_key:
            from tulpar_ai.rag.embed import JinaEmbedder

            embed = JinaEmbedder().embed  # one batch; 429/5xx are retried with backoff in rag/embed.py _post
        qvecs, sent = await embed_with_cache([g["question"] for g in golden], cache=VectorCache(data_dir / "emb_cache"),
                                             embed=embed)
        if qvecs.shape[1] != corpus.shape[1]:
            raise SystemExit(f"question vectors have {qvecs.shape[1]} dims, the index {corpus.shape[1]}")
        rng = np.random.default_rng(args.seed)
        hit_rank = _hit_rank_fn()
        rows = [analyze_question(g, q, corpus, payloads, hit_rank=hit_rank, rng=rng, n_random=args.n_random,
                                 top_n=s.rag_top_n) for g, q in zip(golden, qvecs)]
        for r, q in zip(rows, qvecs):
            r["qdrant_top4_match"] = qdrant_top_ids(client, collection, q, s.rag_top_n) == [t["id"] for t in r["top"]]
    finally:
        client.close()
    c2, q2, meta = project(corpus, qvecs, args.method, args.seed, fit=args.umap_fit)
    meta["neighbour_preservation"] = neighbour_preservation(corpus, qvecs, c2, q2)
    src = [p.get("source") for p in payloads]
    report = {
        "note": "Real vectors of the production index; the questions are real golden questions (not synthetic). "
                "Cosines are in the original 1024-d space; the 2-D map is an illustration and distorts distances.",
        "collection": collection, "embed_model": EMBED_MODEL, "query_task": QUERY_TASK, "passage_task": "retrieval.passage",
        "dim": int(corpus.shape[1]), "chunks": len(payloads), "chunks_by_source": {k: src.count(k) for k in SOURCES},
        "retrieval": {"dense_top_n": s.rag_top_n, "rerank": "off (RAG_RERANK=auto for Jina)", "rewrite": "not applied",
                      "hit_rule": "evals/run.py _hit_rank (WHO page ±1)", "rag_min_score": s.rag_min_score},
        "random_baseline": {"n": args.n_random, "seed": args.seed, "pool": "corpus minus the question's expected chunks"},
        "projection": meta, "api": {"texts_embedded_this_run": sent, "corpus_reembedded": False},
        "summary": summarize(rows, s.rag_top_n),
        "eval_cross_check": compare_with_eval(rows, Path(args.check_results), s.rag_top_n),
        "questions": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "embedding_map.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if not args.no_plots:
        plot_map(c2, q2, payloads, rows, meta, out)
        plot_bars(rows, s.rag_min_score, args.n_random, args.seed, s.rag_top_n, out)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--copy-from", help="embedded Qdrant folder to copy into AI_DATA_DIR/qdrant (the .lock is skipped)")
    ap.add_argument("--collection", help="default: the collection the app opens for the current corpus")
    ap.add_argument("--golden", default=str(GOLDEN))
    ap.add_argument("--ids", default="", help="comma-separated golden ids (default: 8 WHO + 2 technique + 2 nutrition)")
    ap.add_argument("--method", choices=("umap", "pca"), default="umap")
    ap.add_argument("--umap-fit", choices=("joint", "transform"), default="joint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--check-results", default=str(ROOT / "evals" / "results" / "qa_final_nutrition_rules.json"),
                    help="saved qa run whose hit ranks must match (missing file: the check is skipped)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--trace", action="store_true", help="send the question embedding call to LangSmith")
    args = ap.parse_args(argv)
    os.environ.setdefault("AI_DATA_DIR", str(ROOT / "data" / "embedding-map"))
    if not args.trace:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    report = asyncio.run(run(args))
    print(json.dumps({k: report[k] for k in ("summary", "eval_cross_check", "api", "projection")}, ensure_ascii=False,
                     indent=1))
    for r in report["questions"]:
        rel = r["relevant"]["score"] if r["relevant"] else None
        print(f"{r['id']} rank={r['hit_rank']} relevant={rel} random={r['random']['mean']} top3="
              + " | ".join(f"{_src_label(t, 30)} {t['score']:.3f}" for t in r["top"][:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
