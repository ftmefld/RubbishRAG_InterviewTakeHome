import json
from pathlib import Path

from rubbish_rag.judge import HeuristicFeatureJudge


DATASET_PATH = Path("data/judge_dataset.json")
SPLIT = "dev"


def main():
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    data = [
        item
        for item in data
        if item.get("split") == SPLIT
    ]

    judge = HeuristicFeatureJudge()

    routing_errors = []
    clarify_errors = []

    print(
        "\nAdaptive Judge Error Analysis"
    )
    print("=" * 110)
    print(f"Split: {SPLIT}")
    print(f"Cases: {len(data)}")
    print()

    for i, item in enumerate(
        data,
        start=1,
    ):
        query = item["query"]

        expected_retrieval = item.get(
            "expected_retrieval",
            "RETRIEVE",
        )

        expected_clarification = item.get(
            "expected_clarification",
            "DO_NOT_CLARIFY",
        )

        decision = judge.pre_retrieval(
            query
        )

        predicted_retrieval = (
            "RETRIEVE"
            if decision.needs_retrieval
            else "NO_RETRIEVAL"
        )

        predicted_clarification = (
            "CLARIFY"
            if decision.needs_clarification
            else "DO_NOT_CLARIFY"
        )

        routing_ok = (
            predicted_retrieval
            == expected_retrieval
        )

        clarify_ok = (
            predicted_clarification
            == expected_clarification
        )

        marker = (
            "OK"
            if routing_ok and clarify_ok
            else "ERROR"
        )

        print(
            f"[{i:02d}] {marker}"
        )

        print(
            f"QUERY: {query}"
        )

        print(
            "  Retrieval:"
            f" expected={expected_retrieval}"
            f" predicted={predicted_retrieval}"
        )

        print(
            "  Clarify:  "
            f" expected={expected_clarification}"
            f" predicted={predicted_clarification}"
        )

        print(
            f"  action={decision.action}"
            f" top_k={decision.top_k}"
            f" confidence={decision.confidence}"
        )

        print(
            "  missing_slots="
            f"{decision.missing_slots}"
        )

        print(
            "  reasons="
            f"{decision.reason_codes}"
        )

        if not routing_ok:
            routing_errors.append(
                {
                    "query": query,
                    "expected": expected_retrieval,
                    "predicted": predicted_retrieval,
                    "decision": decision.to_dict(),
                }
            )

        if not clarify_ok:
            clarify_errors.append(
                {
                    "query": query,
                    "expected": expected_clarification,
                    "predicted": predicted_clarification,
                    "decision": decision.to_dict(),
                }
            )

        print("-" * 110)

    print()
    print("=" * 110)

    print(
        f"Routing errors: "
        f"{len(routing_errors)}"
    )

    for item in routing_errors:
        print(
            f"  - {item['query']}"
        )

        print(
            f"    expected="
            f"{item['expected']}, "
            f"predicted="
            f"{item['predicted']}"
        )

    print()

    print(
        f"Clarification errors: "
        f"{len(clarify_errors)}"
    )

    for item in clarify_errors:
        print(
            f"  - {item['query']}"
        )

        print(
            f"    expected="
            f"{item['expected']}, "
            f"predicted="
            f"{item['predicted']}"
        )

        print(
            f"    missing_slots="
            f"{item['decision']['missing_slots']}"
        )

        print(
            f"    reasons="
            f"{item['decision']['reason_codes']}"
        )


if __name__ == "__main__":
    main()