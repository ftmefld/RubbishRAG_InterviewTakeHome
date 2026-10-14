import copy
import json
import statistics
import time
from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline  # registers candidate pipeline stages

from server.evaluator import score_one, aggregate


BENCH_PATH = Path("server/visible_bench.json")
OUTPUT_PATH = Path(
    "analysis/full_candidate_pipeline_results.json"
)

STOCK_METRICS_PATH = Path(
    "traces/metrics.json"
)

FINAL_TOPK = 5
DIAGNOSTIC_TOPK = 20


def same_doc_id(a, b):
    return str(a) == str(b)


def rank_of(hits, gold_doc_id):
    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        if same_doc_id(
            hit.get("doc_id"),
            gold_doc_id,
        ):
            return rank

    return None


def hit_at(rank, k):
    return (
        rank is not None
        and rank <= k
    )


def percentile(values, q):
    if not values:
        return 0.0

    values = sorted(values)

    index = max(
        0,
        min(
            len(values) - 1,
            int(
                q * len(values)
            ) - 1,
        ),
    )

    return values[index]


def load_stock_metrics():
    if not STOCK_METRICS_PATH.exists():
        return None

    try:
        with STOCK_METRICS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return data.get("agg")

    except Exception:
        return None


def main():
    with BENCH_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        benchmark = json.load(f)

    retriever = R.get("retriever")
    reranker = R.get("reranker")
    resolver = R.get("resolver")

    score_rows = []
    detail_rows = []
    durations = []

    retrieval_hit1 = 0
    retrieval_hit3 = 0
    retrieval_hit5 = 0
    retrieval_hit20 = 0

    reranked_hit1 = 0
    reranked_hit3 = 0
    reranked_hit5 = 0

    improved = 0
    regressed = 0
    unchanged = 0

    print(
        "\nFull Candidate Pipeline Evaluation"
    )
    print("=" * 96)

    for item in benchmark:
        query = item["query"]
        gold = item["gold_doc_id"]

        t0 = time.perf_counter()

        # Important:
        # ask our registered retriever for 20 candidates once.
        #
        # Our current retriever already retrieves a candidate
        # pool of 20 internally, so this lets us measure both:
        #   - final Top-5 behavior
        #   - diagnostic Recall@20
        #
        # without a second retrieval call.
        retrieved_20 = retriever(
            query,
            topk=DIAGNOSTIC_TOPK,
        )

        retrieval_rank20 = rank_of(
            retrieved_20,
            gold,
        )

        retrieved_5 = [
            dict(hit)
            for hit in retrieved_20[:FINAL_TOPK]
        ]

        retrieval_rank5 = rank_of(
            retrieved_5,
            gold,
        )

        reranked = reranker(
            query,
            copy.deepcopy(retrieved_5),
        )

        reranked_rank = rank_of(
            reranked,
            gold,
        )

        prediction = resolver(
            query,
            reranked,
        )

        duration_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        durations.append(
            duration_ms
        )

        score = score_one(
            prediction,
            item,
        )

        score_rows.append(
            {
                **score,
                "id": item["id"],
            }
        )

        retrieval_hit1 += int(
            hit_at(
                retrieval_rank5,
                1,
            )
        )

        retrieval_hit3 += int(
            hit_at(
                retrieval_rank5,
                3,
            )
        )

        retrieval_hit5 += int(
            hit_at(
                retrieval_rank5,
                5,
            )
        )

        retrieval_hit20 += int(
            hit_at(
                retrieval_rank20,
                20,
            )
        )

        reranked_hit1 += int(
            hit_at(
                reranked_rank,
                1,
            )
        )

        reranked_hit3 += int(
            hit_at(
                reranked_rank,
                3,
            )
        )

        reranked_hit5 += int(
            hit_at(
                reranked_rank,
                5,
            )
        )

        if (
            retrieval_rank5 is not None
            and reranked_rank is not None
        ):
            if reranked_rank < retrieval_rank5:
                rank_change = "improved"
                improved += 1

            elif reranked_rank > retrieval_rank5:
                rank_change = "regressed"
                regressed += 1

            else:
                rank_change = "unchanged"
                unchanged += 1

        elif (
            retrieval_rank5 is None
            and reranked_rank is None
        ):
            rank_change = "unchanged"
            unchanged += 1

        else:
            # This normally should not happen because reranking
            # receives exactly the retrieved Top-5.
            rank_change = "changed"

        cites = prediction.get(
            "cites",
            [],
        )

        gold_cited = any(
            same_doc_id(
                cite,
                gold,
            )
            for cite in cites[:3]
        )

        detail = {
            "id": item["id"],
            "query": query,
            "expect": item.get("expect"),
            "gold_doc_id": gold,

            "retrieval_rank_top20": (
                retrieval_rank20
            ),

            "retrieval_rank_top5": (
                retrieval_rank5
            ),

            "reranked_gold_rank": (
                reranked_rank
            ),

            "rank_change": (
                rank_change
            ),

            "prediction_status": (
                prediction.get("status")
            ),

            "cites": cites,

            "gold_cited_top3": (
                gold_cited
            ),

            "score": score,

            "retrieved_top5": [
                hit.get("doc_id")
                for hit in retrieved_5
            ],

            "reranked_top5": [
                hit.get("doc_id")
                for hit in reranked
            ],

            "latency_ms": round(
                duration_ms,
                2,
            ),
        }

        detail_rows.append(
            detail
        )

        print(
            f"{item['id']:<5} "
            f"R20={str(retrieval_rank20):<4} "
            f"R5={str(retrieval_rank5):<4} "
            f"RR={str(reranked_rank):<4} "
            f"status="
            f"{prediction.get('status', '?'):<8} "
            f"cited={str(gold_cited):<5} "
            f"{rank_change}"
        )

    total = len(benchmark)

    final_metrics = aggregate(
        score_rows
    )

    final_metrics["timing_ms"] = {
        "mean": round(
            statistics.mean(durations),
            1,
        ),
        "p50": round(
            statistics.median(durations),
            1,
        ),
        "p95": round(
            percentile(
                durations,
                0.95,
            ),
            1,
        ),
    }

    retrieval_metrics = {
        "hit_at_1": (
            retrieval_hit1 / total
        ),
        "hit_at_3": (
            retrieval_hit3 / total
        ),
        "hit_at_5": (
            retrieval_hit5 / total
        ),
        "recall_at_20": (
            retrieval_hit20 / total
        ),
    }

    reranking_metrics = {
        "hit_at_1": (
            reranked_hit1 / total
        ),
        "hit_at_3": (
            reranked_hit3 / total
        ),
        "hit_at_5": (
            reranked_hit5 / total
        ),
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
    }

    stock_metrics = (
        load_stock_metrics()
    )

    print("\nRetrieval")
    print("-" * 96)

    print(
        f"Candidate Hit@1  : "
        f"{retrieval_hit1}/{total} "
        f"({100 * retrieval_hit1 / total:.1f}%)"
    )

    print(
        f"Candidate Hit@3  : "
        f"{retrieval_hit3}/{total} "
        f"({100 * retrieval_hit3 / total:.1f}%)"
    )

    print(
        f"Candidate Hit@5  : "
        f"{retrieval_hit5}/{total} "
        f"({100 * retrieval_hit5 / total:.1f}%)"
    )

    print(
        f"Candidate Recall@20 : "
        f"{retrieval_hit20}/{total} "
        f"({100 * retrieval_hit20 / total:.1f}%)"
    )

    print("\nAfter reranking")
    print("-" * 96)

    print(
        f"Reranked Hit@1 : "
        f"{reranked_hit1}/{total} "
        f"({100 * reranked_hit1 / total:.1f}%)"
    )

    print(
        f"Reranked Hit@3 : "
        f"{reranked_hit3}/{total} "
        f"({100 * reranked_hit3 / total:.1f}%)"
    )

    print(
        f"Reranked Hit@5 : "
        f"{reranked_hit5}/{total} "
        f"({100 * reranked_hit5 / total:.1f}%)"
    )

    print(
        f"Improved={improved} "
        f"Regressed={regressed} "
        f"Unchanged={unchanged}"
    )

    print("\nEnd-to-end scoring")
    print("-" * 96)

    print(
        json.dumps(
            final_metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    if stock_metrics:
        print(
            "\nStock bench vs full candidate pipeline"
        )
        print("-" * 96)

        print(
            f"Stock accuracy     : "
            f"{stock_metrics.get('accuracy')}"
        )

        print(
            f"Candidate accuracy : "
            f"{final_metrics.get('accuracy')}"
        )

        print(
            f"Stock answer_acc   : "
            f"{stock_metrics.get('answer_acc')}"
        )

        print(
            f"Candidate answer_acc: "
            f"{final_metrics.get('answer_acc')}"
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
                "total": total,
                "retrieval": (
                    retrieval_metrics
                ),
                "reranking": (
                    reranking_metrics
                ),
                "final": (
                    final_metrics
                ),
                "stock_visible_bench": (
                    stock_metrics
                ),
                "details": (
                    detail_rows
                ),
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