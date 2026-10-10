import json
from pathlib import Path

import matplotlib.pyplot as plt


LENGTH_PATH = Path(
    "analysis/task_b_length_results.json"
)

RRF_PATH = Path(
    "analysis/task_b_rrf_results.json"
)

GATED_PATH = Path(
    "analysis/task_b_gated_rrf_results.json"
)

OUTPUT_PATH = Path(
    "plots/length_vs_bm25.png"
)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    length_data = load_json(LENGTH_PATH)
    rrf_data = load_json(RRF_PATH)
    gated_data = load_json(GATED_PATH)

    gold_rows = length_data["gold_documents"]

    # --------------------------------------------------
    # Left panel:
    # Does longer gold evidence receive worse BM25 rank?
    # --------------------------------------------------

    length_ratios = []
    rank_gaps = []
    labels = []

    for row in gold_rows:
        if (
            row["length_ratio"] is None
            or row["bm25_rank"] is None
            or row["dense_rank"] is None
        ):
            continue

        length_ratios.append(
            row["length_ratio"]
        )

        # Positive:
        # BM25 ranks gold worse than dense.
        rank_gaps.append(
            row["bm25_rank"]
            - row["dense_rank"]
        )

        labels.append(row["id"])

    # --------------------------------------------------
    # Right panel:
    # Baseline vs global RRF vs gated RRF
    # --------------------------------------------------

    rrf_summary = rrf_data["summary"]

    baseline_hit3 = (
        gated_data["baseline"]["hit3"]
    )

    baseline_hit5 = (
        gated_data["baseline"]["hit5"]
    )

    global_rrf_hit3 = (
        rrf_summary["rrf_hit3"]
    )

    global_rrf_hit5 = (
        rrf_summary["rrf_hit5"]
    )

    gated_hit3 = (
        gated_data["gated_rrf"]["hit3"]
    )

    gated_hit5 = (
        gated_data["gated_rrf"]["hit5"]
    )

    methods = [
        "Server\nfusion",
        "Global\nRRF",
        "Gated\nRRF",
    ]

    hit3_values = [
        baseline_hit3,
        global_rrf_hit3,
        gated_hit3,
    ]

    hit5_values = [
        baseline_hit5,
        global_rrf_hit5,
        gated_hit5,
    ]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
    )

    # ----- Length diagnostic -----

    ax = axes[0]

    ax.scatter(
        length_ratios,
        rank_gaps,
        s=55,
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    for x, y, label in zip(
        length_ratios,
        rank_gaps,
        labels,
    ):
        ax.annotate(
            label,
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(
        "Gold length / median candidate length"
    )

    ax.set_ylabel(
        "BM25 rank − dense rank"
    )

    ax.set_title(
        "Length hypothesis: no systematic BM25 penalty"
    )

    ax.text(
        0.02,
        0.98,
        "Upper-right = long gold ranked worse by BM25",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    # ----- Fusion ablation -----

    ax = axes[1]

    x = list(range(len(methods)))
    width = 0.34

    ax.bar(
        [i - width / 2 for i in x],
        hit3_values,
        width=width,
        label="Hit@3",
    )

    ax.bar(
        [i + width / 2 for i in x],
        hit5_values,
        width=width,
        label="Hit@5",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.set_ylim(0, 0.7)

    ax.set_ylabel("Recall")

    ax.set_title(
        "Fusion ablation"
    )

    ax.legend()

    for i, value in enumerate(hit3_values):
        ax.text(
            i - width / 2,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )

    for i, value in enumerate(hit5_values):
        ax.text(
            i + width / 2,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )

    fig.suptitle(
        "Task B — Length Diagnostics and Hybrid Fusion Ablation",
        fontsize=13,
    )

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved plot to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()