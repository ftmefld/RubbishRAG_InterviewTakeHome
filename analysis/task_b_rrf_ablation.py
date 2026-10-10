import json
from pathlib import Path
from statistics import mean, median

from server._hidden_retriever import remote_retrieve


BENCH_PATH = Path("server/visible_bench.json")
OUTPUT_PATH = Path("analysis/task_b_rrf_results.json")

CANDIDATE_POOL = 20
FINAL_K = 5
RRF_K = 60


def load_benchmark():
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_candidate_ids(hits):
    """
    A document can contribute multiple chunks with the same doc_id.
    RRF must rank individual candidates/chunks, not collapse them by doc_id.
    """
    result = []

    for candidate_id, hit in enumerate(hits):
        item = dict(hit)
        item["_candidate_id"] = candidate_id
        result.append(item)

    return result


def rank_map(hits, score_key):
    """
    Map each unique candidate/chunk to its 1-based rank.
    """
    ranked = sorted(
        hits,
        key=lambda hit: hit.get(score_key, float("-inf")),
        reverse=True,
    )

    return {
        hit["_candidate_id"]: rank
        for rank, hit in enumerate(ranked, start=1)
    }


def reciprocal_rank_fusion(hits):
    bm25_ranks = rank_map(hits, "bm25")
    dense_ranks = rank_map(hits, "dense")

    rescored = []

    for hit in hits:
        candidate_id = hit["_candidate_id"]

        bm25_rank = bm25_ranks[candidate_id]
        dense_rank = dense_ranks[candidate_id]

        rrf_score = (
            1.0 / (RRF_K + bm25_rank)
            + 1.0 / (RRF_K + dense_rank)
        )

        new_hit = dict(hit)
        new_hit["bm25_rank"] = bm25_rank
        new_hit["dense_rank"] = dense_rank
        new_hit["rrf_score"] = rrf_score

        rescored.append(new_hit)

    # Use original fused score only as a deterministic tie-breaker.
    return sorted(
        rescored,
        key=lambda hit: (
            hit["rrf_score"],
            hit.get("fused", 0.0),
        ),
        reverse=True,
    )


def find_gold_rank(hits, gold_doc_id):
    """
    Evaluation is document-level.

    If multiple chunks belong to the gold document,
    the best-ranked occurrence counts.
    """
    for rank, hit in enumerate(hits, start=1):
        if hit["doc_id"] == gold_doc_id:
            return rank

    return None


def hit_at(rank, k):
    return rank is not None and rank <= k


def format_rank(rank):
    return str(rank) if rank is not None else "MISS"


def main():
    benchmark = load_benchmark()
    results = []

    print("\nTask B — Baseline Fusion vs RRF")
    print("=" * 100)

    print(
        f"{'ID':<5}"
        f"{'FusedRank':<12}"
        f"{'BM25Rank':<11}"
        f"{'DenseRank':<12}"
        f"{'RRFRank':<10}"
        f"{'Delta':<8}"
        f"{'RRF@5'}"
    )

    print("-" * 100)

    for example in benchmark:
        query_id = example["id"]
        query = example["query"]
        gold_doc_id = example["gold_doc_id"]

        raw_hits = remote_retrieve(
            query,
            topk=CANDIDATE_POOL,
        )

        hits = add_candidate_ids(raw_hits)

        # Server's original fused order
        fused_rank = find_gold_rank(hits, gold_doc_id)

        bm25_sorted = sorted(
            hits,
            key=lambda hit: hit.get("bm25", float("-inf")),
            reverse=True,
        )

        dense_sorted = sorted(
            hits,
            key=lambda hit: hit.get("dense", float("-inf")),
            reverse=True,
        )

        bm25_rank = find_gold_rank(
            bm25_sorted,
            gold_doc_id,
        )

        dense_rank = find_gold_rank(
            dense_sorted,
            gold_doc_id,
        )

        rrf_hits = reciprocal_rank_fusion(hits)

        rrf_rank = find_gold_rank(
            rrf_hits,
            gold_doc_id,
        )

        if fused_rank is not None and rrf_rank is not None:
            delta = fused_rank - rrf_rank
        else:
            delta = None

        results.append(
            {
                "id": query_id,
                "query": query,
                "gold_doc_id": gold_doc_id,
                "fused_rank": fused_rank,
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank,
                "rrf_rank": rrf_rank,
                "delta": delta,
                "fused_hit3": hit_at(fused_rank, 3),
                "fused_hit5": hit_at(fused_rank, 5),
                "rrf_hit3": hit_at(rrf_rank, 3),
                "rrf_hit5": hit_at(rrf_rank, 5),
                "top5_fused": [
                    {
                        "doc_id": hit["doc_id"],
                        "bm25": hit.get("bm25"),
                        "dense": hit.get("dense"),
                        "fused": hit.get("fused"),
                    }
                    for hit in hits[:FINAL_K]
                ],
                "top5_rrf": [
                    {
                        "doc_id": hit["doc_id"],
                        "bm25_rank": hit["bm25_rank"],
                        "dense_rank": hit["dense_rank"],
                        "rrf_score": hit["rrf_score"],
                    }
                    for hit in rrf_hits[:FINAL_K]
                ],
            }
        )

        delta_display = (
            f"{delta:+d}"
            if delta is not None
            else "-"
        )

        print(
            f"{query_id:<5}"
            f"{format_rank(fused_rank):<12}"
            f"{format_rank(bm25_rank):<11}"
            f"{format_rank(dense_rank):<12}"
            f"{format_rank(rrf_rank):<10}"
            f"{delta_display:<8}"
            f"{hit_at(rrf_rank, 5)}"
        )

    n = len(results)

    fused_hit3 = sum(
        row["fused_hit3"] for row in results
    ) / n

    fused_hit5 = sum(
        row["fused_hit5"] for row in results
    ) / n

    rrf_hit3 = sum(
        row["rrf_hit3"] for row in results
    ) / n

    rrf_hit5 = sum(
        row["rrf_hit5"] for row in results
    ) / n

    comparable = [
        row
        for row in results
        if row["fused_rank"] is not None
        and row["rrf_rank"] is not None
    ]

    improved = [
        row
        for row in comparable
        if row["rrf_rank"] < row["fused_rank"]
    ]

    regressed = [
        row
        for row in comparable
        if row["rrf_rank"] > row["fused_rank"]
    ]

    unchanged = [
        row
        for row in comparable
        if row["rrf_rank"] == row["fused_rank"]
    ]

    deltas = [
        row["delta"]
        for row in comparable
    ]

    summary = {
        "candidate_pool": CANDIDATE_POOL,
        "rrf_k": RRF_K,
        "queries": n,
        "server_fused_hit3": fused_hit3,
        "server_fused_hit5": fused_hit5,
        "rrf_hit3": rrf_hit3,
        "rrf_hit5": rrf_hit5,
        "improved_queries": [
            row["id"] for row in improved
        ],
        "regressed_queries": [
            row["id"] for row in regressed
        ],
        "unchanged_queries": [
            row["id"] for row in unchanged
        ],
        "mean_rank_delta": (
            mean(deltas) if deltas else None
        ),
        "median_rank_delta": (
            median(deltas) if deltas else None
        ),
    }

    print("\nAggregate comparison")
    print("=" * 50)

    print(
        f"Server fused Hit@3 : "
        f"{fused_hit3:.3f} ({fused_hit3 * 100:.1f}%)"
    )

    print(
        f"RRF Hit@3          : "
        f"{rrf_hit3:.3f} ({rrf_hit3 * 100:.1f}%)"
    )

    print(
        f"Server fused Hit@5 : "
        f"{fused_hit5:.3f} ({fused_hit5 * 100:.1f}%)"
    )

    print(
        f"RRF Hit@5          : "
        f"{rrf_hit5:.3f} ({rrf_hit5 * 100:.1f}%)"
    )

    print(
        "\nImproved  :",
        [row["id"] for row in improved],
    )

    print(
        "Regressed :",
        [row["id"] for row in regressed],
    )

    print(
        "Unchanged :",
        [row["id"] for row in unchanged],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "summary": summary,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved detailed results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()