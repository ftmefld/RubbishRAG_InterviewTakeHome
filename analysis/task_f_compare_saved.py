import json
from pathlib import Path


BEFORE_PATH = Path(
    "analysis/task_f_poison_baseline.json"
)

AFTER_PATH = Path(
    "analysis/task_f_post_eval_results.json"
)


def is_poison(doc_id):
    """
    Challenge label used only for evaluation.
    Production code does not use this rule.
    """
    try:
        return int(doc_id) >= 9000
    except (TypeError, ValueError):
        return False


def main():
    with BEFORE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        before = json.load(f)

    with AFTER_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        after = json.load(f)

    before_poison = 0
    before_total = 0
    before_queries = 0

    for row in before["queries"]:
        poison_in_query = False

        for hit in row["top3"]:
            before_total += 1

            if is_poison(hit["doc_id"]):
                before_poison += 1
                poison_in_query = True

        if poison_in_query:
            before_queries += 1

    after_poison = (
        after["poison_top3_after"]
    )

    after_total = (
        after["total_top3"]
    )

    after_queries = (
        after["queries_with_poison_after"]
    )

    print(
        "\nTrack F — Before/After Comparison"
    )
    print("=" * 65)

    print("\nBefore Track F")
    print("-" * 65)

    print(
        f"Queries with poison Top-3 : "
        f"{before_queries}/20"
    )

    print(
        f"Poison citations          : "
        f"{before_poison}/{before_total} "
        f"({100 * before_poison / before_total:.1f}%)"
    )

    print(
        f"Gold Top-1                : "
        f"{before['top1_rate'] * 100:.1f}%"
    )

    print(
        f"Gold Hit@3                : "
        f"{before['hit3'] * 100:.1f}%"
    )

    print("\nAfter Track F")
    print("-" * 65)

    print(
        f"Queries with poison Top-3 : "
        f"{after_queries}/20"
    )

    print(
        f"Poison citations          : "
        f"{after_poison}/{after_total} "
        f"({100 * after_poison / after_total:.1f}%)"
    )

    print(
        f"Gold Top-1                : "
        f"{after['reranked_top1_rate'] * 100:.1f}%"
    )

    print(
        f"Gold Hit@3                : "
        f"{after['reranked_hit3'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()