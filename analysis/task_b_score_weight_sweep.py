import json
from pathlib import Path

from server._hidden_retriever import remote_retrieve


BENCH_PATH = Path("server/visible_bench.json")
OUTPUT_PATH = Path("analysis/task_b_score_weight_results.json")

CANDIDATE_POOL = 20

WEIGHTS = [
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


def load_benchmark():
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize(score, max_score):
    if max_score <= 0:
        return 0.0

    return score / max_score


def weighted_score_fusion(hits, bm25_weight):
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
        bm25_norm = normalize(
            hit.get("bm25", 0.0),
            max_bm25,
        )

        dense_norm = normalize(
            hit.get("dense", 0.0),
            max_dense,
        )

        score = (
            bm25_weight * bm25_norm
            + (1.0 - bm25_weight) * dense_norm
        )

        item = dict(hit)
        item["weighted_score"] = score
        rescored.append(item)

    return sorted(
        rescored,
        key=lambda h: (
            h["weighted_score"],
            h.get("fused", 0.0),
        ),
        reverse=True,
    )


def gold_rank(hits, gold_doc_id):
    for rank, hit in enumerate(hits, start=1):
        if hit["doc_id"] == gold_doc_id:
            return rank

    return None


def hit_at(rank, k):
    return rank is not None and rank <= k


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    return 1.0 / rank


def main():
    benchmark = load_benchmark()

    # Retrieve once per query.
    query_data = []

    for example in benchmark:
        hits = remote_retrieve(
            example["query"],
            topk=CANDIDATE_POOL,
        )

        query_data.append(
            {
                "example": example,
                "hits": hits,
            }
        )

    n = len(query_data)

    # --------------------------------
    # Verify w=0.5 reproduces server
    # --------------------------------

    exact_top5 = 0

    for item in query_data:
        reconstructed = weighted_score_fusion(
            item["hits"],
            bm25_weight=0.5,
        )

        server_ids = [
            hit["doc_id"]
            for hit in item["hits"][:5]
        ]

        reconstructed_ids = [
            hit["doc_id"]
            for hit in reconstructed[:5]
        ]

        if server_ids == reconstructed_ids:
            exact_top5 += 1

    print("\nTask B — Raw Score Weight Sensitivity")
    print("=" * 80)

    print(
        f"Baseline reconstruction at w=0.50: "
        f"{exact_top5}/{n} exact Top-5"
    )

    print("\nWeight sweep")
    print("-" * 80)

    results = []

    for weight in WEIGHTS:
        h3 = 0
        h5 = 0
        mrr = 0.0

        ranks = {}

        for item in query_data:
            example = item["example"]

            ranked = weighted_score_fusion(
                item["hits"],
                bm25_weight=weight,
            )

            rank = gold_rank(
                ranked,
                example["gold_doc_id"],
            )

            ranks[example["id"]] = rank

            h3 += hit_at(rank, 3)
            h5 += hit_at(rank, 5)
            mrr += reciprocal_rank(rank)

        result = {
            "bm25_weight": weight,
            "dense_weight": 1.0 - weight,
            "hit3": h3 / n,
            "hit5": h5 / n,
            "mrr": mrr / n,
            "ranks": ranks,
        }

        results.append(result)

        print(
            f"BM25={weight:.2f} "
            f"Dense={1.0 - weight:.2f}  "
            f"Hit@3={h3 / n:.3f}  "
            f"Hit@5={h5 / n:.3f}  "
            f"MRR={mrr / n:.3f}"
        )

    # Compare every setting against baseline 0.5
    baseline = next(
        result
        for result in results
        if result["bm25_weight"] == 0.50
    )

    print("\nChanges relative to baseline w=0.50")
    print("-" * 80)

    for result in results:
        if result["bm25_weight"] == 0.50:
            continue

        improved = []
        regressed = []

        for query_id, baseline_rank in baseline["ranks"].items():
            new_rank = result["ranks"][query_id]

            if baseline_rank is None and new_rank is not None:
                improved.append(query_id)

            elif baseline_rank is not None and new_rank is None:
                regressed.append(query_id)

            elif (
                baseline_rank is not None
                and new_rank is not None
            ):
                if new_rank < baseline_rank:
                    improved.append(query_id)

                elif new_rank > baseline_rank:
                    regressed.append(query_id)

        print(
            f"\nBM25={result['bm25_weight']:.2f}"
        )
        print("  Improved :", improved)
        print("  Regressed:", regressed)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "baseline_reconstruction": {
                    "exact_top5": exact_top5,
                    "total": n,
                },
                "results": results,
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