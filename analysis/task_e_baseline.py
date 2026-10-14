import json
from pathlib import Path


RESULTS_PATH = Path(
    "analysis/full_candidate_pipeline_results.json"
)

OUTPUT_PATH = Path(
    "analysis/task_e_baseline_results.json"
)


def duplicate_count(doc_ids):
    """
    Number of redundant slots beyond the first occurrence
    of each document.
    """
    return len(doc_ids) - len(set(doc_ids))


def has_duplicate(doc_ids):
    return len(doc_ids) != len(set(doc_ids))


def unique_ratio(doc_ids):
    if not doc_ids:
        return 1.0

    return len(set(doc_ids)) / len(doc_ids)


def main():
    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        results = json.load(f)

    details = results["details"]

    total_queries = len(details)

    retrieved_dup_queries = 0
    reranked_dup_queries = 0

    retrieved_duplicate_slots = 0
    reranked_duplicate_slots = 0

    retrieved_total_slots = 0
    reranked_total_slots = 0

    retrieved_unique_top3 = []
    reranked_unique_top3 = []

    rows = []

    print("\nTrack E — Diversity Baseline")
    print("=" * 90)

    for row in details:
        retrieved = row["retrieved_top5"]
        reranked = row["reranked_top5"]

        retrieved_dups = duplicate_count(
            retrieved
        )

        reranked_dups = duplicate_count(
            reranked
        )

        retrieved_total_slots += len(
            retrieved
        )

        reranked_total_slots += len(
            reranked
        )

        retrieved_duplicate_slots += (
            retrieved_dups
        )

        reranked_duplicate_slots += (
            reranked_dups
        )

        retrieved_dup_queries += int(
            has_duplicate(retrieved)
        )

        reranked_dup_queries += int(
            has_duplicate(reranked)
        )

        retrieved_top3 = retrieved[:3]
        reranked_top3 = reranked[:3]

        retrieved_unique_top3.append(
            len(set(retrieved_top3))
        )

        reranked_unique_top3.append(
            len(set(reranked_top3))
        )

        if (
            retrieved_dups > 0
            or reranked_dups > 0
        ):
            print(
                f"{row['id']:<5} "
                f"retrieved={retrieved} "
                f"reranked={reranked}"
            )

        rows.append(
            {
                "id": row["id"],
                "retrieved_top5": retrieved,
                "reranked_top5": reranked,
                "retrieved_duplicate_slots": (
                    retrieved_dups
                ),
                "reranked_duplicate_slots": (
                    reranked_dups
                ),
                "retrieved_top3_unique_docs": (
                    len(set(retrieved_top3))
                ),
                "reranked_top3_unique_docs": (
                    len(set(reranked_top3))
                ),
            }
        )

    retrieved_slot_rate = (
        retrieved_duplicate_slots
        / retrieved_total_slots
        if retrieved_total_slots
        else 0.0
    )

    reranked_slot_rate = (
        reranked_duplicate_slots
        / reranked_total_slots
        if reranked_total_slots
        else 0.0
    )

    avg_retrieved_top3_unique = (
        sum(retrieved_unique_top3)
        / len(retrieved_unique_top3)
    )

    avg_reranked_top3_unique = (
        sum(reranked_unique_top3)
        / len(reranked_unique_top3)
    )

    print("\nRetrieved Top-5")
    print("-" * 90)

    print(
        f"Queries with duplicate documents : "
        f"{retrieved_dup_queries}/{total_queries} "
        f"({100 * retrieved_dup_queries / total_queries:.1f}%)"
    )

    print(
        f"Duplicate slot rate              : "
        f"{retrieved_duplicate_slots}/"
        f"{retrieved_total_slots} "
        f"({100 * retrieved_slot_rate:.1f}%)"
    )

    print(
        f"Average unique docs in Top-3     : "
        f"{avg_retrieved_top3_unique:.2f}/3"
    )

    print("\nCurrent reranked Top-5")
    print("-" * 90)

    print(
        f"Queries with duplicate documents : "
        f"{reranked_dup_queries}/{total_queries} "
        f"({100 * reranked_dup_queries / total_queries:.1f}%)"
    )

    print(
        f"Duplicate slot rate              : "
        f"{reranked_duplicate_slots}/"
        f"{reranked_total_slots} "
        f"({100 * reranked_slot_rate:.1f}%)"
    )

    print(
        f"Average unique docs in Top-3     : "
        f"{avg_reranked_top3_unique:.2f}/3"
    )

    print("\nCurrent ranking quality")
    print("-" * 90)

    reranking = results["reranking"]

    print(
        f"Hit@1 : "
        f"{100 * reranking['hit_at_1']:.1f}%"
    )

    print(
        f"Hit@3 : "
        f"{100 * reranking['hit_at_3']:.1f}%"
    )

    print(
        f"Hit@5 : "
        f"{100 * reranking['hit_at_5']:.1f}%"
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
                "retrieved": {
                    "queries_with_duplicates": (
                        retrieved_dup_queries
                    ),
                    "duplicate_slot_rate": (
                        retrieved_slot_rate
                    ),
                    "average_unique_top3": (
                        avg_retrieved_top3_unique
                    ),
                },
                "reranked": {
                    "queries_with_duplicates": (
                        reranked_dup_queries
                    ),
                    "duplicate_slot_rate": (
                        reranked_slot_rate
                    ),
                    "average_unique_top3": (
                        avg_reranked_top3_unique
                    ),
                },
                "hit_at_1": reranking[
                    "hit_at_1"
                ],
                "hit_at_3": reranking[
                    "hit_at_3"
                ],
                "hit_at_5": reranking[
                    "hit_at_5"
                ],
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