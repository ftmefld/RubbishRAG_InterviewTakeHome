import json
from pathlib import Path

from server._hidden_retriever import remote_retrieve


BENCH_PATH = Path("server/visible_bench.json")
OUTPUT_PATH = Path("analysis/task_b_gated_rrf_results.json")

CANDIDATE_POOL = 20
RRF_K = 60


def load_benchmark():
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def quantile(values, q):
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


def has_high_outlier(values):
    """
    Tukey upper-outlier rule:

        max > Q3 + 1.5 * IQR

    This is distribution-based and does not use benchmark labels.
    """
    if len(values) < 4:
        return False

    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)

    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr

    return max(values) > upper_fence


def add_candidate_ids(hits):
    result = []

    for candidate_id, hit in enumerate(hits):
        item = dict(hit)
        item["_candidate_id"] = candidate_id
        result.append(item)

    return result


def rank_map(hits, score_key):
    ranked = sorted(
        hits,
        key=lambda h: h.get(score_key, float("-inf")),
        reverse=True,
    )

    return {
        hit["_candidate_id"]: rank
        for rank, hit in enumerate(ranked, start=1)
    }


def rrf_fusion(hits):
    bm25_ranks = rank_map(hits, "bm25")
    dense_ranks = rank_map(hits, "dense")

    rescored = []

    for hit in hits:
        candidate_id = hit["_candidate_id"]

        score = (
            1.0
            / (RRF_K + bm25_ranks[candidate_id])
            +
            1.0
            / (RRF_K + dense_ranks[candidate_id])
        )

        item = dict(hit)

        item["rrf_score"] = score
        item["bm25_rank"] = bm25_ranks[candidate_id]
        item["dense_rank"] = dense_ranks[candidate_id]

        rescored.append(item)

    return sorted(
        rescored,
        key=lambda h: (
            h["rrf_score"],
            h.get("fused", 0.0),
        ),
        reverse=True,
    )


def should_use_rrf(hits):
    """
    Activate rank fusion only when:

    1. BM25 or dense score distribution contains a high outlier.
    2. BM25 and dense disagree on the top candidate.

    This targets the max-normalization failure mode while avoiding
    unnecessary reranking when both retrieval signals already agree.
    """
    bm25_scores = [
        hit.get("bm25", 0.0)
        for hit in hits
    ]

    dense_scores = [
        hit.get("dense", 0.0)
        for hit in hits
    ]

    bm25_outlier = has_high_outlier(
        bm25_scores
    )

    dense_outlier = has_high_outlier(
        dense_scores
    )

    bm25_top = max(
        hits,
        key=lambda h: h.get(
            "bm25",
            float("-inf"),
        ),
    )["_candidate_id"]

    dense_top = max(
        hits,
        key=lambda h: h.get(
            "dense",
            float("-inf"),
        ),
    )["_candidate_id"]

    disagreement = (
        bm25_top != dense_top
    )

    use_rrf = (
        (bm25_outlier or dense_outlier)
        and disagreement
    )

    diagnostics = {
        "bm25_outlier": bm25_outlier,
        "dense_outlier": dense_outlier,
        "top1_disagreement": disagreement,
        "use_rrf": use_rrf,
    }

    return use_rrf, diagnostics


def gated_fusion(hits):
    use_rrf, diagnostics = should_use_rrf(
        hits
    )

    if use_rrf:
        ranked = rrf_fusion(hits)
        strategy = "rrf"
    else:
        # Preserve the server's original fused ranking.
        ranked = list(hits)
        strategy = "server_fused"

    return ranked, strategy, diagnostics


def gold_rank(hits, gold_doc_id):
    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        if hit["doc_id"] == gold_doc_id:
            return rank

    return None


def hit_at(rank, k):
    return (
        rank is not None
        and rank <= k
    )


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    return 1.0 / rank


def main():
    benchmark = load_benchmark()

    rows = []

    baseline_h3 = 0
    baseline_h5 = 0
    baseline_rr = 0.0

    gated_h3 = 0
    gated_h5 = 0
    gated_rr = 0.0

    rrf_used = 0

    print(
        "\nTask B — Gated RRF Ablation"
    )
    print("=" * 100)

    print(
        f"{'ID':<5}"
        f"{'Base':<8}"
        f"{'Gated':<8}"
        f"{'Branch':<16}"
        f"{'BM25Out':<10}"
        f"{'DenseOut':<10}"
        f"{'Disagree'}"
    )

    print("-" * 100)

    for example in benchmark:
        raw_hits = remote_retrieve(
            example["query"],
            topk=CANDIDATE_POOL,
        )

        hits = add_candidate_ids(
            raw_hits
        )

        baseline_rank = gold_rank(
            hits,
            example["gold_doc_id"],
        )

        gated_hits, strategy, diagnostics = (
            gated_fusion(hits)
        )

        gated_rank = gold_rank(
            gated_hits,
            example["gold_doc_id"],
        )

        baseline_h3 += hit_at(
            baseline_rank,
            3,
        )

        baseline_h5 += hit_at(
            baseline_rank,
            5,
        )

        baseline_rr += reciprocal_rank(
            baseline_rank
        )

        gated_h3 += hit_at(
            gated_rank,
            3,
        )

        gated_h5 += hit_at(
            gated_rank,
            5,
        )

        gated_rr += reciprocal_rank(
            gated_rank
        )

        if strategy == "rrf":
            rrf_used += 1

        rows.append(
            {
                "id": example["id"],
                "query": example["query"],
                "gold_doc_id": example[
                    "gold_doc_id"
                ],
                "baseline_rank": baseline_rank,
                "gated_rank": gated_rank,
                "strategy": strategy,
                **diagnostics,
            }
        )

        base_text = (
            str(baseline_rank)
            if baseline_rank is not None
            else "MISS"
        )

        gated_text = (
            str(gated_rank)
            if gated_rank is not None
            else "MISS"
        )

        print(
            f"{example['id']:<5}"
            f"{base_text:<8}"
            f"{gated_text:<8}"
            f"{strategy:<16}"
            f"{str(diagnostics['bm25_outlier']):<10}"
            f"{str(diagnostics['dense_outlier']):<10}"
            f"{diagnostics['top1_disagreement']}"
        )

    n = len(benchmark)

    print("\nAggregate")
    print("=" * 60)

    print(
        f"Baseline Hit@3 : "
        f"{baseline_h3 / n:.3f}"
    )

    print(
        f"Gated Hit@3    : "
        f"{gated_h3 / n:.3f}"
    )

    print()

    print(
        f"Baseline Hit@5 : "
        f"{baseline_h5 / n:.3f}"
    )

    print(
        f"Gated Hit@5    : "
        f"{gated_h5 / n:.3f}"
    )

    print()

    print(
        f"Baseline MRR   : "
        f"{baseline_rr / n:.3f}"
    )

    print(
        f"Gated MRR      : "
        f"{gated_rr / n:.3f}"
    )

    print()

    print(
        f"RRF activated  : "
        f"{rrf_used}/{n} queries"
    )

    improved = []
    regressed = []
    unchanged = []

    for row in rows:
        base = (
            row["baseline_rank"]
            if row["baseline_rank"]
            is not None
            else float("inf")
        )

        gated = (
            row["gated_rank"]
            if row["gated_rank"]
            is not None
            else float("inf")
        )

        if gated < base:
            improved.append(row["id"])

        elif gated > base:
            regressed.append(row["id"])

        else:
            unchanged.append(row["id"])

    print(
        "\nImproved :",
        improved,
    )

    print(
        "Regressed:",
        regressed,
    )

    print(
        "Unchanged:",
        unchanged,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "baseline": {
                    "hit3": baseline_h3 / n,
                    "hit5": baseline_h5 / n,
                    "mrr": baseline_rr / n,
                },
                "gated_rrf": {
                    "hit3": gated_h3 / n,
                    "hit5": gated_h5 / n,
                    "mrr": gated_rr / n,
                    "rrf_activated": rrf_used,
                },
                "improved": improved,
                "regressed": regressed,
                "unchanged": unchanged,
                "queries": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()