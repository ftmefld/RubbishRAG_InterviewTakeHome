import json
from collections import Counter
from pathlib import Path
from statistics import mean


RESULTS_PATH = Path(
    "analysis/task_b_score_weight_results.json"
)

OUTPUT_PATH = Path(
    "analysis/task_b_weight_cv_results.json"
)

BASELINE_WEIGHT = 0.50
N_FOLDS = 5


def hit_at(rank, k):
    return rank is not None and rank <= k


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    return 1.0 / rank


def metrics(query_ids, ranks):
    """
    Compute retrieval metrics for a subset of query IDs.
    """
    n = len(query_ids)

    if n == 0:
        return {
            "hit3": 0.0,
            "hit5": 0.0,
            "mrr": 0.0,
        }

    hit3 = 0
    hit5 = 0
    rr_sum = 0.0

    for query_id in query_ids:
        rank = ranks.get(query_id)

        hit3 += hit_at(rank, 3)
        hit5 += hit_at(rank, 5)
        rr_sum += reciprocal_rank(rank)

    return {
        "hit3": hit3 / n,
        "hit5": hit5 / n,
        "mrr": rr_sum / n,
    }


def selection_key(result, train_ids):
    """
    Select using:
      1. Hit@5
      2. Hit@3
      3. MRR
      4. If still tied, prefer the weight closest to baseline 0.50

    The final rule is deliberately conservative.
    """
    m = metrics(
        train_ids,
        result["ranks"],
    )

    distance_from_baseline = abs(
        result["bm25_weight"] - BASELINE_WEIGHT
    )

    return (
        m["hit5"],
        m["hit3"],
        m["mrr"],
        -distance_from_baseline,
    )


def main():
    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    results = data["results"]

    # Convert string JSON keys if necessary.
    for result in results:
        result["bm25_weight"] = float(
            result["bm25_weight"]
        )

    baseline = next(
        result
        for result in results
        if result["bm25_weight"] == BASELINE_WEIGHT
    )

    # All configurations contain the same query IDs.
    query_ids = list(
        baseline["ranks"].keys()
    )

    # Deterministic 5-fold split:
    # fold 0: v0, v5, v10, v15
    # fold 1: v1, v6, v11, v16
    # ...
    folds = [
        query_ids[i::N_FOLDS]
        for i in range(N_FOLDS)
    ]

    print("\nTask B — Weight Cross-Validation")
    print("=" * 88)

    print(
        f"Queries: {len(query_ids)}"
    )

    print(
        f"Folds: {N_FOLDS} "
        f"({len(query_ids) // N_FOLDS} held-out queries each)"
    )

    print(
        "\nSelection rule: "
        "Hit@5 → Hit@3 → MRR → closest to baseline"
    )

    fold_results = []

    # These hold one genuinely held-out prediction
    # for every query.
    cv_selected_ranks = {}
    cv_baseline_ranks = {}

    print("\nFold results")
    print("-" * 88)

    for fold_index, test_ids in enumerate(
        folds,
        start=1,
    ):
        train_ids = [
            query_id
            for query_id in query_ids
            if query_id not in test_ids
        ]

        selected = max(
            results,
            key=lambda result: selection_key(
                result,
                train_ids,
            ),
        )

        selected_weight = selected[
            "bm25_weight"
        ]

        train_selected_metrics = metrics(
            train_ids,
            selected["ranks"],
        )

        test_selected_metrics = metrics(
            test_ids,
            selected["ranks"],
        )

        test_baseline_metrics = metrics(
            test_ids,
            baseline["ranks"],
        )

        for query_id in test_ids:
            cv_selected_ranks[query_id] = (
                selected["ranks"][query_id]
            )

            cv_baseline_ranks[query_id] = (
                baseline["ranks"][query_id]
            )

        fold_result = {
            "fold": fold_index,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "selected_weight": selected_weight,
            "train_selected_metrics": (
                train_selected_metrics
            ),
            "test_selected_metrics": (
                test_selected_metrics
            ),
            "test_baseline_metrics": (
                test_baseline_metrics
            ),
        }

        fold_results.append(fold_result)

        print(
            f"\nFold {fold_index}"
        )

        print(
            f"  Test queries     : {test_ids}"
        )

        print(
            f"  Selected weight  : "
            f"BM25={selected_weight:.2f}, "
            f"Dense={1.0 - selected_weight:.2f}"
        )

        print(
            "  Held-out selected: "
            f"H3={test_selected_metrics['hit3']:.3f} "
            f"H5={test_selected_metrics['hit5']:.3f} "
            f"MRR={test_selected_metrics['mrr']:.3f}"
        )

        print(
            "  Held-out baseline: "
            f"H3={test_baseline_metrics['hit3']:.3f} "
            f"H5={test_baseline_metrics['hit5']:.3f} "
            f"MRR={test_baseline_metrics['mrr']:.3f}"
        )

    # ------------------------------------
    # Pooled held-out CV performance
    # ------------------------------------

    selected_cv_metrics = metrics(
        query_ids,
        cv_selected_ranks,
    )

    baseline_cv_metrics = metrics(
        query_ids,
        cv_baseline_ranks,
    )

    selected_weights = [
        result["selected_weight"]
        for result in fold_results
    ]

    frequency = Counter(selected_weights)

    print("\n" + "=" * 88)
    print("Pooled held-out results")
    print("=" * 88)

    print(
        "Cross-validated selected weights:"
    )

    for weight, count in sorted(
        frequency.items()
    ):
        print(
            f"  BM25={weight:.2f}: "
            f"{count}/{N_FOLDS} folds"
        )

    print(
        "\nAdaptive BM25-heavy selection:"
    )

    print(
        f"  Hit@3 = "
        f"{selected_cv_metrics['hit3']:.3f}"
    )

    print(
        f"  Hit@5 = "
        f"{selected_cv_metrics['hit5']:.3f}"
    )

    print(
        f"  MRR   = "
        f"{selected_cv_metrics['mrr']:.3f}"
    )

    print(
        "\nBaseline 0.50 / 0.50:"
    )

    print(
        f"  Hit@3 = "
        f"{baseline_cv_metrics['hit3']:.3f}"
    )

    print(
        f"  Hit@5 = "
        f"{baseline_cv_metrics['hit5']:.3f}"
    )

    print(
        f"  MRR   = "
        f"{baseline_cv_metrics['mrr']:.3f}"
    )

    # Per-query held-out comparison.
    improved = []
    regressed = []
    unchanged = []

    for query_id in query_ids:
        selected_rank = cv_selected_ranks[
            query_id
        ]

        baseline_rank = cv_baseline_ranks[
            query_id
        ]

        selected_value = (
            selected_rank
            if selected_rank is not None
            else float("inf")
        )

        baseline_value = (
            baseline_rank
            if baseline_rank is not None
            else float("inf")
        )

        if selected_value < baseline_value:
            improved.append(query_id)

        elif selected_value > baseline_value:
            regressed.append(query_id)

        else:
            unchanged.append(query_id)

    print("\nHeld-out rank changes")
    print("-" * 88)

    print(
        "Improved :",
        improved,
    )

    print(
        "Regressed:",
        regressed,
    )

    print(
        "Unchanged:",
        unchanged,
    )

    output = {
        "selection_rule": (
            "Hit@5 -> Hit@3 -> MRR -> "
            "closest to baseline"
        ),
        "baseline_weight": BASELINE_WEIGHT,
        "folds": fold_results,
        "selected_weight_frequency": {
            str(weight): count
            for weight, count in frequency.items()
        },
        "pooled_cv_selected": (
            selected_cv_metrics
        ),
        "pooled_cv_baseline": (
            baseline_cv_metrics
        ),
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()