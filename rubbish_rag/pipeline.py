"""RubbishRAG pipeline — the 1-minute version.

Stages (override any of them with @rubbish_rag.* decorators —
see docs/DECORATORS.md for the mechanism):
  chunk -> retrieve (remote) -> rerank -> resolve
"""
import rubbish_rag as R
from .normalize_fa import normalize_fa
from .slots import extract_slots


@R.chunker
def chunk(docs):
    """Current behavior: splits Answer into fixed 300-char pieces.

    TODO: read a few long Answers, then look at what comes out here.
    Can every chunk below be judged on its own? What is missing?
    You decide the chunking strategy.
    """
    chunks = []
    for d in docs:
        text = (d.get("Answer") or d.get("BriefAnswer") or "")
        for i in range(0, len(text), 300):
            part = text[i:i + 300]
            if part.strip():
                chunks.append({
                    "doc_id": d["doc_id"],
                    "category": d.get("Category", ""),
                    "text": part,
                    "chunk_id": f'{d["doc_id"]}:{i // 300}',
                })
    return chunks

def _quantile(values, q):
    values = sorted(values)

    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower] * (1.0 - fraction)
        + values[upper] * fraction
    )


def _has_high_outlier(values):
    """
    Detect an upper score outlier with Tukey's standard 1.5*IQR rule.
    """
    if len(values) < 4:
        return False

    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)

    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr

    return max(values) > upper_fence


def _rank_map(hits, score_key):
    ranked_ids = sorted(
        range(len(hits)),
        key=lambda i: hits[i].get(
            score_key,
            float("-inf"),
        ),
        reverse=True,
    )

    return {
        candidate_id: rank
        for rank, candidate_id in enumerate(
            ranked_ids,
            start=1,
        )
    }


def _rrf_fusion(hits, k=60):
    """
    Equal-weight reciprocal rank fusion over BM25 and dense rankings.
    """
    bm25_ranks = _rank_map(
        hits,
        "bm25",
    )

    dense_ranks = _rank_map(
        hits,
        "dense",
    )

    rescored = []

    for candidate_id, hit in enumerate(hits):
        item = dict(hit)

        item["rrf_score"] = (
            1.0 / (k + bm25_ranks[candidate_id])
            + 1.0 / (k + dense_ranks[candidate_id])
        )

        rescored.append(item)

    return sorted(
        rescored,
        key=lambda hit: (
            hit["rrf_score"],
            hit.get("fused", 0.0),
        ),
        reverse=True,
    )


def _should_use_rrf(hits):
    if not hits:
        return False

    bm25_scores = [
        hit.get("bm25", 0.0)
        for hit in hits
    ]

    dense_scores = [
        hit.get("dense", 0.0)
        for hit in hits
    ]

    has_outlier = (
        _has_high_outlier(bm25_scores)
        or _has_high_outlier(dense_scores)
    )

    bm25_top = max(
        range(len(hits)),
        key=lambda i: hits[i].get(
            "bm25",
            float("-inf"),
        ),
    )

    dense_top = max(
        range(len(hits)),
        key=lambda i: hits[i].get(
            "dense",
            float("-inf"),
        ),
    )

    top1_disagreement = (
        bm25_top != dense_top
    )

    return (
        has_outlier
        and top1_disagreement
    )

@R.retriever
def retrieve(query, topk=5):
    """
    Retrieve a broader server candidate pool and apply RRF only when
    score-outlier distortion coincides with BM25/dense disagreement.

    Otherwise preserve the server's original fused ranking.
    """
    from server._hidden_retriever import remote_retrieve

    candidate_pool = max(
        20,
        topk,
    )

    hits = remote_retrieve(
        query,
        topk=candidate_pool,
    )

    if not hits:
        return []

    if _should_use_rrf(hits):
        hits = _rrf_fusion(hits)

    return hits[:topk]


@R.reranker
def rerank(query, hits):
    """
    Normalize Persian text before lexical overlap scoring so equivalent
    digit scripts, Arabic/Persian letter variants, half-spaces, and common
    colloquial forms do not create token-level false negatives.
    """
    normalized_query = normalize_fa(query)

    query_tokens = set(
        normalized_query.split()
    )

    for hit in hits:
        normalized_text = normalize_fa(
            hit["text"]
        )

        hit_tokens = set(
            normalized_text.split()
        )

        hit["rerank_score"] = len(
            query_tokens & hit_tokens
        )

    return sorted(
        hits,
        key=lambda hit: hit["rerank_score"],
        reverse=True,
    )

@R.resolver
def resolve(query, hits):
    """Current behavior: always returns an answer built from the top hits.

    TODO: what should happen when the top hits contradict each other, or when
    the query itself is missing the one detail the answer depends on?
    Check the output contract in the README, then decide.
    """
    slots = extract_slots(query)  
    top = hits[0] if hits else {"text": "", "doc_id": -1}
    return {"status": "answer", "brief": top["text"][:200],
            "cites": [h["doc_id"] for h in hits[:3]] or [top["doc_id"]]}
