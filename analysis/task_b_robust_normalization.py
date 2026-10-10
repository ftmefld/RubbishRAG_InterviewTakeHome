import json
from pathlib import Path

from server._hidden_retriever import remote_retrieve


BENCH_PATH = Path("server/visible_bench.json")
OUTPUT_PATH = Path("analysis/task_b_robust_normalization_results.json")

CANDIDATE_POOL = 20

# 1.00 = normal max normalization
PERCENTILES = [0.75, 0.90, 0.95, 1.00]


def load_benchmark():
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_candidate_ids(hits):
    """Give every returned chunk a unique identity."""
    result = []

    for candidate_id, hit in enumerate(hits):
        item = dict(hit)
        item["_candidate_id"] = candidate_id
        result.append(item)

    return result


def quantile(values, q):
    """
    Linear-interpolated quantile without NumPy.
    q must be between 0 and 1.
    """
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


def clipped_normalize(score, denominator):
    """
    Normalize to [0, 1].

    Scores above the percentile threshold are clipped to 1,
    preventing a single extreme maximum from compressing
    the rest of the candidate distribution.
    """
    if denominator <= 0:
        return 0.0

    value = score / denominator

    return max(
        0.0,
        min(value, 1.0),
    )


def robust_fusion(hits, percentile):
    bm25_scores = [
        hit.get("bm25", 0.0)
        for hit in hits
    ]

    dense_scores = [
        hit.get("dense", 0.0)
        for hit in hits
    ]

    bm25_denominator = quantile(
        bm25_scores,
        percentile,
    )

    dense_denominator = quantile(
        dense_scores,
        percentile,
    )

    rescored = []

    for hit in hits:
        bm25_norm = clipped_normalize(
            hit.get("bm25", 0.0),
            bm25_denominator,
        )

        dense_norm = clipped_normalize(
            hit.get("dense", 0.0),
            dense_denominator,
        )

        robust_score = (
            0.5 * bm25_norm
            + 0.5 * dense_norm
        )

        item = dict(hit)

        item["bm25_robust_norm"] = bm25_norm
        item["dense_robust_norm"] = dense_norm
        item["robust_score"] = robust_score

        rescored.append(item)

    return sorted(
        rescored,
        key=lambda hit: (
            hit["robust_score"],
            hit.get("fused", 0.0),
        ),
        reverse=True,
    )


def reconstructed_max_fusion(hits):
    """
    Reconstruct the stated naive formula:

        0.5 * BM25/max(BM25)
      + 0.5 * dense/max(dense)

    Used only as a sanity check.
    """
    max_bm25 = max(
        (hit.get("bm25", 0.0) for hit in hits),
        default=0.0,
    )

    max_dense = max(
        (hit.get("dense", 0.0) for hit in hits),
        default=0.0,
    )

    rescored = []

    for hit in hits:
        bm25_norm = clipped_normalize(
            hit.get("bm25", 0.0),
            max_bm25,
        )

        dense_norm = clipped_normalize(
            hit.get("dense", 0.0),
            max_dense,
        )

        score = (
            0.5 * bm25_norm
            + 0.5 * dense_norm
        )

        item = dict(hit)
        item["reconstructed_score"] = score

        rescored.append(item)

    return sorted(
        rescored,
        key=lambda hit: (
            hit["reconstructed_score"],
            hit.get("fused", 0.0),
        ),
        reverse=True,
    )


def gold_rank(hits, gold_doc_id):
    for rank, hit in enumerate(hits, start=1):
        if hit["doc_id"] == gold_doc_id:
            return rank

    return None


def hit_at(rank, k):
    return (
        rank is not None
        and rank <= k
    )


def main():
    benchmark = load_benchmark()

    # Retrieve exactly once per query.
    query_data = []

    for example in benchmark:
        raw_hits = remote_retrieve(
            example["query"],
            topk=CANDIDATE_POOL,
        )

        query_data.append(
            {
                "example": example,
                "hits": add_candidate_ids(raw_hits),
            }
        )

    n = len(query_data)

    # -----------------------------------------
    # Baseline: server fused order
    # -----------------------------------------

    server_h3 = 0
    server_h5 = 0

    server_ranks = {}

    for item in query_data:
        example = item["example"]

        rank = gold_rank(
            item["hits"],
            example["gold_doc_id"],
        )

        server_ranks[example["id"]] = rank

        server_h3 += hit_at(rank, 3)
        server_h5 += hit_at(rank, 5)

    print("\nTask B — Robust Score Normalization")
    print("=" * 78)

    print(
        f"Server fused baseline: "
        f"Hit@3={server_h3 / n:.3f} "
        f"Hit@5={server_h5 / n:.3f}"
    )

    # -----------------------------------------
    # Sanity check:
    # Can we approximately reproduce server fusion?
    # -----------------------------------------

    exact_top5_matches = 0

    for item in query_data:
        server_top5 = [
            hit["_candidate_id"]
            for hit in item["hits"][:5]
        ]

        reconstructed = reconstructed_max_fusion(
            item["hits"]
        )

        reconstructed_top5 = [
            hit["_candidate_id"]
            for hit in reconstructed[:5]
        ]

        if server_top5 == reconstructed_top5:
            exact_top5_matches += 1

    print(
        f"Exact Top-5 ordering reproduced by stated "
        f"max-normalization formula: "
        f"{exact_top5_matches}/{n}"
    )

    # -----------------------------------------
    # Percentile normalization sweep
    # -----------------------------------------

    print("\nPercentile normalization")
    print("-" * 78)

    all_results = []

    for percentile in PERCENTILES:
        h3 = 0
        h5 = 0

        improved = []
        regressed = []
        unchanged = []

        per_query = []

        for item in query_data:
            example = item["example"]

            ranked = robust_fusion(
                item["hits"],
                percentile=percentile,
            )

            rank = gold_rank(
                ranked,
                example["gold_doc_id"],
            )

            original_rank = server_ranks[
                example["id"]
            ]

            h3 += hit_at(rank, 3)
            h5 += hit_at(rank, 5)

            if (
                original_rank is not None
                and rank is not None
            ):
                if rank < original_rank:
                    improved.append(example["id"])

                elif rank > original_rank:
                    regressed.append(example["id"])

                else:
                    unchanged.append(example["id"])

            per_query.append(
                {
                    "id": example["id"],
                    "server_rank": original_rank,
                    "robust_rank": rank,
                }
            )

        result = {
            "percentile": percentile,
            "hit3": h3 / n,
            "hit5": h5 / n,
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "per_query": per_query,
        }

        all_results.append(result)

        print(
            f"P={percentile:.2f}  "
            f"Hit@3={h3 / n:.3f}  "
            f"Hit@5={h5 / n:.3f}  "
            f"Improved={len(improved):2d}  "
            f"Regressed={len(regressed):2d}"
        )

    print("\nDetailed changes")
    print("-" * 78)

    for result in all_results:
        print(
            f"\nP={result['percentile']:.2f}"
        )

        print(
            "  Improved :",
            result["improved"],
        )

        print(
            "  Regressed:",
            result["regressed"],
        )

    output = {
        "server_baseline": {
            "hit3": server_h3 / n,
            "hit5": server_h5 / n,
        },
        "reconstruction": {
            "exact_top5_matches": exact_top5_matches,
            "total_queries": n,
        },
        "percentile_results": all_results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()