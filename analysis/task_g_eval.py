import json
from pathlib import Path

from rubbish_rag.pipeline import _polarity_conflicts


OUTPUT_PATH = Path(
    "analysis/task_g_eval_results.json"
)


CASES = [
    # --------------------------------------------------
    # True polarity conflicts
    # --------------------------------------------------
    {
        "id": "c1",
        "query": "شرکت سابقه تاخیر ندارد",
        "candidate": "شرکت سابقه تاخیر دارد",
        "expect_conflict": True,
    },
    {
        "id": "c2",
        "query": "شرکت بدهی ندارد",
        "candidate": "شرکت بدهی دارد",
        "expect_conflict": True,
    },
    {
        "id": "c3",
        "query": "چک در گزارش ثبت نشده",
        "candidate": "چک در گزارش ثبت شده است",
        "expect_conflict": True,
    },
    {
        "id": "c4",
        "query": "شرکت فاقد سابقه مالیاتی است",
        "candidate": "شرکت سابقه مالیاتی دارد",
        "expect_conflict": True,
    },

    # --------------------------------------------------
    # Same polarity / compatible
    # --------------------------------------------------
    {
        "id": "s1",
        "query": "شرکت سابقه تاخیر ندارد",
        "candidate": "شرکت هیچ سابقه تاخیری ندارد",
        "expect_conflict": False,
    },
    {
        "id": "s2",
        "query": "شرکت بدهی ندارد",
        "candidate": "شرکت فاقد بدهی است",
        "expect_conflict": False,
    },

    # --------------------------------------------------
    # Neutral / not a factual inversion
    # --------------------------------------------------
    {
        "id": "n1",
        "query": (
            "نمی خواهم چک برگشتی "
            "در سابقه ام باشد"
        ),
        "candidate": (
            "چک برگشتی روی امتیاز "
            "اعتباری تاثیر دارد"
        ),
        "expect_conflict": False,
    },
    {
        "id": "n2",
        "query": (
            "شرکت سابقه تاخیر ندارد، "
            "چرا امتیازش بالا نیست؟"
        ),
        "candidate": (
            "عوامل مختلفی مانند مالیات، "
            "تسهیلات و چک بر امتیاز اثر دارند"
        ),
        "expect_conflict": False,
    },
]


def main():
    correct = 0
    false_positive = 0
    false_negative = 0

    rows = []

    print(
        "\nTrack G — Polarity Evaluation"
    )
    print("=" * 82)

    for case in CASES:
        conflicts = _polarity_conflicts(
            case["query"],
            case["candidate"],
        )

        predicted_conflict = bool(
            conflicts
        )

        expected_conflict = case[
            "expect_conflict"
        ]

        passed = (
            predicted_conflict
            == expected_conflict
        )

        correct += int(passed)

        if (
            predicted_conflict
            and not expected_conflict
        ):
            false_positive += 1

        if (
            not predicted_conflict
            and expected_conflict
        ):
            false_negative += 1

        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "candidate": (
                    case["candidate"]
                ),
                "expected_conflict": (
                    expected_conflict
                ),
                "predicted_conflict": (
                    predicted_conflict
                ),
                "conflicting_concepts": (
                    conflicts
                ),
                "passed": passed,
            }
        )

        print(
            f"{case['id']:<4} "
            f"expected="
            f"{str(expected_conflict):<5} "
            f"predicted="
            f"{str(predicted_conflict):<5} "
            f"conflicts={conflicts} "
            f"{'OK' if passed else 'FAIL'}"
        )

    total = len(CASES)

    accuracy = (
        correct / total
    )

    print("\nSummary")
    print("-" * 82)

    print(
        f"Polarity classification accuracy : "
        f"{correct}/{total} "
        f"({100 * accuracy:.1f}%)"
    )

    print(
        f"False positives                  : "
        f"{false_positive}"
    )

    print(
        f"False negatives                  : "
        f"{false_negative}"
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
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "false_positive": (
                    false_positive
                ),
                "false_negative": (
                    false_negative
                ),
                "cases": rows,
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