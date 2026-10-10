import json
from pathlib import Path

from server._hidden_retriever import remote_retrieve


BENCH_PATH = Path("server/visible_bench.json")

CANDIDATE_POOL = 20
RRF_K = 60

# Small, pre-defined sensitivity sweep.
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_benchmark():
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def weighted_rrf(hits, bm25_weight):
    bm25_ranks = rank_map(hits, "bm25")
    dense_ranks = rank_map(hits, "dense")

    rescored = []

    for hit in hits:
        cid = hit["_candidate_id"]

        bm25_rank = bm25_ranks[cid]
        dense_rank = dense_ranks[cid]

        score = (
            bm25_weight / (RRF_K + bm25_rank)
            + (1.0 - bm25_weight) / (RRF_K + dense_rank)
        )

        item = dict(hit)
        item["wrrf_score"] = score
        rescored.append(item)

    return sorted(
        rescored,
        key=lambda h: (
            h["wrrf_score"],
            h.get("fused", 0.0),
        ),
        reverse=True,
    )


def gold_rank(hits, gold_doc_id):
    for rank, hit in enumerate(hits, start=1):
        if hit["doc_id"] == gold_doc_id:
            return rank

    return None


def hit(rank, k):
    return rank is not None and rank <= k


def main():
    benchmark = load_benchmark()

    # Retrieve only once per query.
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

    print("\nTask B — Weighted RRF Sensitivity")
    print("=" * 70)

    # Server fused baseline.
    fused_hit3 = 0
    fused_hit5 = 0

    for item in query_data:
        gold = item["example"]["gold_doc_id"]

        rank = gold_rank(
            item["hits"],
            gold,
        )

        fused_hit3 += hit(rank, 3)
        fused_hit5 += hit(rank, 5)

    n = len(query_data)

    print(
        f"Server fused : "
        f"Hit@3={fused_hit3 / n:.3f}  "
        f"Hit@5={fused_hit5 / n:.3f}"
    )

    print("\nWeighted RRF")
    print("-" * 70)

    results = []

    for weight in WEIGHTS:
        h3 = 0
        h5 = 0
        ranks = []

        for item in query_data:
            example = item["example"]

            ranked = weighted_rrf(
                item["hits"],
                bm25_weight=weight,
            )

            rank = gold_rank(
                ranked,
                example["gold_doc_id"],
            )

            ranks.append(
                {
                    "id": example["id"],
                    "rank": rank,
                }
            )

            h3 += hit(rank, 3)
            h5 += hit(rank, 5)

        result = {
            "bm25_weight": weight,
            "dense_weight": 1.0 - weight,
            "hit3": h3 / n,
            "hit5": h5 / n,
            "ranks": ranks,
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
            "hit3": fused_hit3 / n,
            "hit5": fused_hit5 / n,
        },
        "weighted_rrf": results,
    }

    path = Path("analysis/task_b_weighted_rrf_results.json")

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()