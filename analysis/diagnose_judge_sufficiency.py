import json
from pathlib import Path

from rubbish_rag.judge import (
    HeuristicFeatureJudge,
)


DATASET_PATH = Path(
    "data/judge_dataset.json"
)


def diagnose(split):
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    data = [
        item
        for item in data
        if item.get("split") == split
    ]

    judge = HeuristicFeatureJudge()

    print()
    print(
        f"SUFFICIENCY DIAGNOSTIC — {split}"
    )
    print("=" * 100)

    correct = 0

    for index, item in enumerate(
        data,
        start=1,
    ):
        query = item["query"]

        expected = item.get(
            "expected_sufficiency",
            "SUFFICIENT",
        )

        pre = judge.pre_retrieval(
            query
        )

        mock_hits = (
            [
                {
                    "doc_id": 1,
                    "fused": 0.85,
                    "text": (
                        "رتبه اعتباری "
                        "در بازه..."
                    ),
                }
            ]
            if pre.needs_retrieval
            else []
        )

        post = judge.post_retrieval(
            query,
            mock_hits,
        )

        predicted = (
            post.sufficiency.upper()
        )

        ok = (
            predicted
            == expected.upper()
        )

        if ok:
            correct += 1

        marker = (
            "OK"
            if ok
            else "ERROR"
        )

        print(
            f"[{index:02d}] {marker}"
        )

        print(
            f"QUERY: {query}"
        )

        print(
            f"  pre_action = {pre.action}"
        )

        print(
            f"  expected   = {expected}"
        )

        print(
            f"  predicted  = {predicted}"
        )

        print(
            f"  next_action= {post.next_action}"
        )

        print(
            f"  reasons    = {post.reason_codes}"
        )

        print("-" * 100)

    print(
        f"\nAccuracy: "
        f"{correct}/{len(data)} "
        f"({100 * correct / len(data):.1f}%)"
    )


if __name__ == "__main__":
    diagnose("dev")
    diagnose("test")