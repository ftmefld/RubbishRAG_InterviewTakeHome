import json
from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline  # register pipeline stages

from server._hidden_retriever import remote_retrieve


BENCH_PATH = Path("server/visible_bench.json")
OUTPUT_PATH = Path("analysis/task_f_post_eval_results.json")

TOPK = 5


def is_known_poison_for_evaluation(hit):
    """
    Challenge-specific label used ONLY for evaluation.

    The production reranker does not use this rule.
    """
    try:
        return int(hit["doc_id"]) >= 9000
    except (KeyError, TypeError, ValueError):
        return False


def gold_rank(hits, gold_doc_id):
    for rank, hit in enumerate(hits, start=1):
        if hit["doc_id"] == gold_doc_id:
            return rank

    return None


def main():
    with BENCH_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        benchmark = json.load(f)

    reranker = R.get("reranker")

    total_queries = len(benchmark)

    raw_poison_hits = 0
    poison_top3_after = 0
    total_top3 = 0

    queries_with_poison_raw = 0
    queries_with_poison_after = 0

    raw_hit3 = 0
    reranked_hit3 = 0

    raw_top1 = 0
    reranked_top1 = 0

    improved = 0
    regressed = 0
    unchanged = 0

    rows = []

    print("\nTrack F — Post-Defense Evaluation")
    print("=" * 78)

    for example in benchmark:
        query = example["query"]
        gold = example["gold_doc_id"]

        raw_hits = remote_retrieve(
            query,
            topk=TOPK,
        )

        reranked = reranker(
            query,
            [dict(hit) for hit in raw_hits],
        )

        raw_poison = [
            hit
            for hit in raw_hits
            if is_known_poison_for_evaluation(hit)
        ]

        reranked_top3 = reranked[:3]

        poison_after = [
            hit
            for hit in reranked_top3
            if is_known_poison_for_evaluation(hit)
        ]

        raw_poison_hits += len(raw_poison)
        poison_top3_after += len(poison_after)
        total_top3 += len(reranked_top3)

        if raw_poison:
            queries_with_poison_raw += 1

        if poison_after:
            queries_with_poison_after += 1

        raw_rank = gold_rank(
            raw_hits,
            gold,
        )

        reranked_rank = gold_rank(
            reranked,
            gold,
        )

        if raw_rank == 1:
            raw_top1 += 1

        if reranked_rank == 1:
            reranked_top1 += 1

        if raw_rank is not None and raw_rank <= 3:
            raw_hit3 += 1

        if (
            reranked_rank is not None
            and reranked_rank <= 3
        ):
            reranked_hit3 += 1

        if (
            raw_rank is not None
            and reranked_rank is not None
        ):
            if reranked_rank < raw_rank:
                change = "improved"
                improved += 1

            elif reranked_rank > raw_rank:
                change = "regressed"
                regressed += 1

            else:
                change = "unchanged"
                unchanged += 1

        else:
            change = "unchanged"
            unchanged += 1

        row = {
            "id": example["id"],
            "gold_doc_id": gold,
            "raw_gold_rank": raw_rank,
            "reranked_gold_rank": reranked_rank,
            "change": change,
            "raw_poison_ids": [
                hit["doc_id"]
                for hit in raw_poison
            ],
            "top3_after": [
                {
                    "doc_id": hit["doc_id"],
                    "verified_corpus": hit.get(
                        "verified_corpus"
                    ),
                }
                for hit in reranked_top3
            ],
            "poison_after_ids": [
                hit["doc_id"]
                for hit in poison_after
            ],
        }

        rows.append(row)

        if raw_poison or poison_after:
            print(
                f"{example['id']:<5} "
                f"raw_poison="
                f"{row['raw_poison_ids']} "
                f"after="
                f"{row['poison_after_ids']} "
                f"gold_rank="
                f"{raw_rank}->{reranked_rank}"
            )

    poison_citation_rate = (
        poison_top3_after / total_top3
        if total_top3
        else 0.0
    )

    print("\nPoison defense")
    print("-" * 78)

    print(
        f"Queries with poison in raw Top-{TOPK} : "
        f"{queries_with_poison_raw}/{total_queries}"
    )

    print(
        f"Known poison hits retrieved          : "
        f"{raw_poison_hits}"
    )

    print(
        f"Queries with poison after reranking  : "
        f"{queries_with_poison_after}/{total_queries}"
    )

    print(
        f"Poison Top-3 citation rate           : "
        f"{poison_top3_after}/{total_top3} "
        f"({poison_citation_rate * 100:.1f}%)"
    )

    print("\nRanking quality")
    print("-" * 78)

    print(
        f"Raw Top-1 gold rate        : "
        f"{raw_top1}/{total_queries} "
        f"({raw_top1 / total_queries * 100:.1f}%)"
    )

    print(
        f"Reranked Top-1 gold rate   : "
        f"{reranked_top1}/{total_queries} "
        f"({reranked_top1 / total_queries * 100:.1f}%)"
    )

    print(
        f"Raw Hit@3                  : "
        f"{raw_hit3}/{total_queries} "
        f"({raw_hit3 / total_queries * 100:.1f}%)"
    )

    print(
        f"Reranked Hit@3             : "
        f"{reranked_hit3}/{total_queries} "
        f"({reranked_hit3 / total_queries * 100:.1f}%)"
    )

    print(
        f"Improved={improved} "
        f"Regressed={regressed} "
        f"Unchanged={unchanged}"
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
                "queries": total_queries,
                "raw_poison_hits": raw_poison_hits,
                "queries_with_poison_raw": (
                    queries_with_poison_raw
                ),
                "queries_with_poison_after": (
                    queries_with_poison_after
                ),
                "poison_top3_after": (
                    poison_top3_after
                ),
                "total_top3": total_top3,
                "poison_citation_rate": (
                    poison_citation_rate
                ),
                "raw_top1_rate": (
                    raw_top1 / total_queries
                ),
                "reranked_top1_rate": (
                    reranked_top1
                    / total_queries
                ),
                "raw_hit3": (
                    raw_hit3 / total_queries
                ),
                "reranked_hit3": (
                    reranked_hit3
                    / total_queries
                ),
                "improved": improved,
                "regressed": regressed,
                "unchanged": unchanged,
                "details": rows,
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