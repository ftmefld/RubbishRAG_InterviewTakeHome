import json
from pathlib import Path


INPUT_PATH = Path(
    "analysis/full_candidate_pipeline_results.json"
)


def diversify(doc_ids):
    seen = set()

    unique = []
    duplicates = []

    for doc_id in doc_ids:
        if doc_id in seen:
            duplicates.append(doc_id)
        else:
            seen.add(doc_id)
            unique.append(doc_id)

    return unique + duplicates


def rank_of(doc_ids, gold):
    for rank, doc_id in enumerate(
        doc_ids,
        start=1,
    ):
        if str(doc_id) == str(gold):
            return rank

    return None


def main():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    before_duplicate_slots = 0
    after_duplicate_slots = 0

    before_duplicate_queries = 0
    after_duplicate_queries = 0

    before_unique_top3 = []
    after_unique_top3 = []

    before_hit3 = 0
    after_hit3 = 0

    improved = 0
    regressed = 0

    rows = data["details"]

    for row in rows:
        before = row["reranked_top5"]
        after = diversify(before)

        gold = row["gold_doc_id"]

        before_dups = (
            len(before)
            - len(set(before))
        )

        after_top5_dups = (
            len(after)
            - len(set(after))
        )

        before_duplicate_slots += (
            before_dups
        )

        after_duplicate_slots += (
            after_top5_dups
        )

        before_duplicate_queries += int(
            before_dups > 0
        )

        after_duplicate_queries += int(
            after_top5_dups > 0
        )

        before_unique_top3.append(
            len(set(before[:3]))
        )

        after_unique_top3.append(
            len(set(after[:3]))
        )

        before_rank = rank_of(
            before,
            gold,
        )

        after_rank = rank_of(
            after,
            gold,
        )

        before_hit3 += int(
            before_rank is not None
            and before_rank <= 3
        )

        after_hit3 += int(
            after_rank is not None
            and after_rank <= 3
        )

        if (
            before_rank is not None
            and after_rank is not None
        ):
            if after_rank < before_rank:
                improved += 1

            elif after_rank > before_rank:
                regressed += 1

        if before != after:
            print(
                f"{row['id']}: "
                f"{before} -> {after}"
            )

    total = len(rows)

    print(
        "\nTrack E — Diversity Post Evaluation"
    )
    print("=" * 72)

    print(
        f"Top-5 duplicate queries before : "
        f"{before_duplicate_queries}/{total}"
    )

    print(
        f"Top-5 duplicate queries after  : "
        f"{after_duplicate_queries}/{total}"
    )

    print(
        f"Duplicate slots before         : "
        f"{before_duplicate_slots}"
    )

    print(
        f"Duplicate slots after          : "
        f"{after_duplicate_slots}"
    )

    print(
        f"Unique docs in Top-3 before    : "
        f"{sum(before_unique_top3) / total:.2f}/3"
    )

    print(
        f"Unique docs in Top-3 after     : "
        f"{sum(after_unique_top3) / total:.2f}/3"
    )

    print(
        f"Hit@3 before                   : "
        f"{before_hit3}/{total} "
        f"({100 * before_hit3 / total:.1f}%)"
    )

    print(
        f"Hit@3 after                    : "
        f"{after_hit3}/{total} "
        f"({100 * after_hit3 / total:.1f}%)"
    )

    print(
        f"Gold ranks improved={improved} "
        f"regressed={regressed}"
    )


if __name__ == "__main__":
    main()