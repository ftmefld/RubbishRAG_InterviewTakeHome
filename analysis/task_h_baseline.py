import json
from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline


OUTPUT_PATH = Path(
    "analysis/task_h_baseline_results.json"
)


CASES = [
    # --------------------------------------------------
    # Ambiguous: should clarify
    # --------------------------------------------------
    {
        "id": "a1",
        "query": "بدهی دارم، روی امتیازم چه تاثیری داره؟",
        "expect": "clarify",
    },
    {
        "id": "a2",
        "query": "بدهی‌ام را پرداخت کردم، کی از گزارشم حذف میشه؟",
        "expect": "clarify",
    },
    {
        "id": "a3",
        "query": "یه بدهی توی سابقه‌ام دارم، چطور پاکش کنم؟",
        "expect": "clarify",
    },
    {
        "id": "a4",
        "query": "شرکتم بدهی داره، چه تاثیری روی رتبه‌اش میذاره؟",
        "expect": "clarify",
    },

    {
        "id": "a5",
        "query": "رتبه C3 دارم، بهم وام میدن؟",
        "expect": "clarify",
    },
    {
        "id": "a6",
        "query": "با رتبه E3 میتونم وام بگیرم؟",
        "expect": "clarify",
    },
    {
        "id": "a7",
        "query": "امتیازم 520 شده، میتونم تسهیلات بگیرم؟",
        "expect": "clarify",
    },
    {
        "id": "a8",
        "query": "رتبه B2 دارم. برای دریافت وام قبولم میکنن؟",
        "expect": "clarify",
    },

    # --------------------------------------------------
    # Specific: should answer
    # --------------------------------------------------
    {
        "id": "s1",
        "query": "بدهی مالیاتی دارم، چه تاثیری روی امتیازم داره؟",
        "expect": "answer",
    },
    {
        "id": "s2",
        "query": "چک برگشتی دارم، چطور رفع سوءاثر کنم؟",
        "expect": "answer",
    },
    {
        "id": "s3",
        "query": "اقساط وامم عقب افتاده، چه تاثیری روی رتبه‌ام داره؟",
        "expect": "answer",
    },
    {
        "id": "s4",
        "query": "شرکت محکومیت مالی دارد، چه اثری روی رتبه دارد؟",
        "expect": "answer",
    },

    {
        "id": "s5",
        "query": "بانک ملی به رتبه C3 وام میده؟",
        "expect": "answer",
    },
    {
        "id": "s6",
        "query": "بانک صادرات به رتبه E3 تسهیلات میده؟",
        "expect": "answer",
    },

    # Visible-bench-style regression guards
    {
        "id": "s7",
        "query": "رتبه C3 یعنی چی؟",
        "expect": "answer",
    },
    {
        "id": "s8",
        "query": "۴۸۰ شدم. چرا؟",
        "expect": "answer",
    },
    {
        "id": "s9",
        "query": "اگر شرکت محکومیت مالی داشته باشد، چقدر رتبه‌اش کاهش پیدا میکنه؟",
        "expect": "answer",
    },
    {
        "id": "s10",
        "query": "نمی‌خوام این چکم که برگشت خورده تو سابقه‌م بیاد، چه طوری حذفش کنم؟",
        "expect": "answer",
    },
]


MOCK_HITS = [
    {
        "doc_id": 1,
        "text": "یک پاسخ نمونه از اطلاعات اعتباری",
        "fused": 0.8,
    },
    {
        "doc_id": 2,
        "text": "اطلاعات تکمیلی اعتباری",
        "fused": 0.7,
    },
]


def main():
    resolver = R.get("resolver")

    correct = 0

    clarify_total = 0
    clarify_correct = 0

    answer_total = 0
    answer_correct = 0

    rows = []

    for case in CASES:
        result = resolver(
            case["query"],
            [dict(hit) for hit in MOCK_HITS],
        )

        predicted = result.get("status")
        expected = case["expect"]

        ok = predicted == expected

        correct += int(ok)

        if expected == "clarify":
            clarify_total += 1
            clarify_correct += int(ok)
        else:
            answer_total += 1
            answer_correct += int(ok)

        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected": expected,
                "predicted": predicted,
                "passed": ok,
                "result": result,
            }
        )

    total = len(CASES)

    print(
        "\nTrack H — Clarification Baseline"
    )
    print("=" * 72)

    print(
        f"Overall status accuracy : "
        f"{correct}/{total} "
        f"({100 * correct / total:.1f}%)"
    )

    print(
        f"Clarification accuracy  : "
        f"{clarify_correct}/{clarify_total} "
        f"({100 * clarify_correct / clarify_total:.1f}%)"
    )

    print(
        f"Specific answer accuracy: "
        f"{answer_correct}/{answer_total} "
        f"({100 * answer_correct / answer_total:.1f}%)"
    )

    print("\nFailures")
    print("-" * 72)

    for row in rows:
        if not row["passed"]:
            print(
                f"{row['id']}: "
                f"expected={row['expected']} "
                f"predicted={row['predicted']} "
                f"query={row['query']}"
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
                "accuracy": correct / total,
                "clarify_accuracy": (
                    clarify_correct / clarify_total
                ),
                "answer_accuracy": (
                    answer_correct / answer_total
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