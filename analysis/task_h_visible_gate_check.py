import json
from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline


BENCH_PATH = Path("server/visible_bench.json")


MOCK_HITS = [
    {
        "doc_id": 1,
        "text": "نمونه پاسخ معتبر",
        "fused": 0.8,
    },
    {
        "doc_id": 2,
        "text": "اطلاعات تکمیلی",
        "fused": 0.7,
    },
]


def main():
    with BENCH_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        bench = json.load(f)

    resolver = R.get("resolver")

    print(
        "\nTrack H — Visible Benchmark Gate Check"
    )
    print("=" * 90)

    false_clarifications = []

    for case in bench:
        result = resolver(
            case["query"],
            [dict(hit) for hit in MOCK_HITS],
        )

        predicted = result.get("status")

        expected = case.get(
            "expect",
            "answer",
        )

        marker = (
            "OK"
            if predicted == expected
            else "FAIL"
        )

        print(
            f"{case['id']:<5} "
            f"expected={expected:<8} "
            f"predicted={predicted:<8} "
            f"{marker}  "
            f"{case['query']}"
        )

        if (
            expected == "answer"
            and predicted == "clarify"
        ):
            false_clarifications.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "result": result,
                }
            )

    print("\nFalse clarifications")
    print("-" * 90)

    if not false_clarifications:
        print("None")

    for row in false_clarifications:
        print(
            f"\n{row['id']}: "
            f"{row['query']}"
        )

        print(
            f"Question: "
            f"{row['result'].get('question')}"
        )

        print(
            f"Options: "
            f"{row['result'].get('options')}"
        )

    print(
        f"\nTotal false clarifications: "
        f"{len(false_clarifications)}"
    )


if __name__ == "__main__":
    main()