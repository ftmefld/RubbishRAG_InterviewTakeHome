import json
from pathlib import Path

from server._hidden_retriever import remote_retrieve


BENCH_PATH = Path("server/visible_bench.json")
OUTPUT_PATH = Path("analysis/task_b_borda_results.json")

CANDIDATE_POOL = 20

WEIGHTS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


def load_benchmark():
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_candidate_ids(hits):
    output = []

    for candidate_id, hit in enumerate(hits):
        item = dict(hit)
        item["_candidate_id"] = candidate_id
        output.append(item)

    return output


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


def borda_fusion(hits, bm25_weight):
    bm25_ranks = rank_map(hits, "bm25")
    dense_ranks = rank_map(hits, "dense")

    n = len(hits)

    rescored = []

    for hit in hits:
        cid = hit["_candidate_id"]

        bm25_rank = bm25_ranks[cid]
        dense_rank = dense_ranks[cid]

        if n > 1:
            bm25_percentile = (
                n - bm25_rank
            ) / (n - 1)

            dense_percentile = (
                n - dense_rank
            ) / (n - 1)

        else:
            bm25_percentile = 1.0
            dense_percentile = 1.0

        score = (
            bm25_weight * bm25_percentile
            + (1.0 - bm25_weight) * dense_percentile
        )

        item = dict(hit)

        item["bm25_rank"] = bm25_rank
        item["dense_rank"] = dense_rank
        item["borda_score"] = score

        rescored.append(item)

    return sorted(
        rescored,
        key=lambda h: (
            h["borda_score"],
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


def main():
    benchmark = load_benchmark()

    # Fetch once per query so every fusion method sees
    # exactly the same candidate pool.
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

    # Original server fusion.
    server_h3 = 0
    server_h5 = 0

    for item in query_data:
        rank = gold_rank(
            item["hits"],
            item["example"]["gold_doc_id"],
        )

        server_h3 += hit_at(rank, 3)
        server_h5 += hit_at(rank, 5)

    print("\nTask B — Borda Rank Fusion")
    print("=" * 72)

    print(
        f"Server fused : "
        f"Hit@3={server_h3 / n:.3f} "
        f"Hit@5={server_h5 / n:.3f}"
    )

    print("\nBorda sensitivity")
    print("-" * 72)

    results = []

    for weight in WEIGHTS:
        h3 = 0
        h5 = 0

        per_query = []

        for item in query_data:
            example = item["example"]

            ranked = borda_fusion(
                item["hits"],
                bm25_weight=weight,
            )

            rank = gold_rank(
                ranked,
                example["gold_doc_id"],
            )

            h3 += hit_at(rank, 3)
            h5 += hit_at(rank, 5)

            per_query.append(
                {
                    "id": example["id"],
                    "rank": rank,
                }
            )

        result = {
            "bm25_weight": weight,
            "dense_weight": 1.0 - weight,
            "hit3": h3 / n,
            "hit5": h5 / n,
            "ranks": per_query,
        }

        results.append(result)

        print(
            f"BM25={weight:.2f} "
            f"Dense={1.0 - weight:.2f}  "
            f"Hit@3={h3 / n:.3f}  "
            f"Hit@5={h5 / n:.3f}"
        )

    output = {
        "server_fused": {
            "hit3": server_h3 / n,
            "hit5": server_h5 / n,
        },
        "borda": results,
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

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()