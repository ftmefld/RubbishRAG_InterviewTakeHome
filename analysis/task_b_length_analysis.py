import json
import math
from pathlib import Path
from statistics import mean, median

from server._hidden_retriever import remote_retrieve


BENCH_PATH = Path("server/visible_bench.json")
OUTPUT_PATH = Path("analysis/task_b_length_results.json")

CANDIDATE_POOL = 20


def load_benchmark():
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def rankdata(values, reverse=False):
    """
    Simple rank assignment for Spearman correlation.
    Ties receive sequential ranks, which is sufficient for this diagnostic.
    """
    order = sorted(
        range(len(values)),
        key=lambda i: values[i],
        reverse=reverse,
    )

    ranks = [0] * len(values)

    for rank, index in enumerate(order, start=1):
        ranks[index] = rank

    return ranks


def pearson(xs, ys):
    if len(xs) < 2:
        return None

    x_mean = mean(xs)
    y_mean = mean(ys)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    )

    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)

    denominator = math.sqrt(x_var * y_var)

    if denominator == 0:
        return None

    return numerator / denominator


def spearman(xs, ys):
    if len(xs) < 2:
        return None

    x_ranks = rankdata(xs)
    y_ranks = rankdata(ys)

    return pearson(x_ranks, y_ranks)


def find_gold_hit(hits, gold_doc_id):
    matches = [
        hit
        for hit in hits
        if hit["doc_id"] == gold_doc_id
    ]

    if not matches:
        return None

    # Use the gold chunk with the highest fused score.
    return max(
        matches,
        key=lambda hit: hit.get("fused", 0.0),
    )


def find_rank(hits, gold_doc_id, score_key):
    ranked = sorted(
        hits,
        key=lambda hit: hit.get(score_key, float("-inf")),
        reverse=True,
    )

    for rank, hit in enumerate(ranked, start=1):
        if hit["doc_id"] == gold_doc_id:
            return rank

    return None


def main():
    benchmark = load_benchmark()

    all_lengths = []
    all_bm25 = []

    query_correlations = []
    gold_rows = []

    print("\nTask B — Length vs BM25 Analysis")
    print("=" * 90)

    for example in benchmark:
        query_id = example["id"]
        query = example["query"]
        gold_doc_id = example["gold_doc_id"]

        hits = remote_retrieve(
            query,
            topk=CANDIDATE_POOL,
        )

        lengths = [
            len(hit.get("text", ""))
            for hit in hits
        ]

        bm25_scores = [
            hit.get("bm25", 0.0)
            for hit in hits
        ]

        all_lengths.extend(lengths)
        all_bm25.extend(bm25_scores)

        corr = spearman(
            lengths,
            bm25_scores,
        )

        if corr is not None:
            query_correlations.append(corr)

        gold_hit = find_gold_hit(
            hits,
            gold_doc_id,
        )

        if gold_hit is None:
            continue

        gold_length = len(
            gold_hit.get("text", "")
        )

        median_candidate_length = median(lengths)

        bm25_rank = find_rank(
            hits,
            gold_doc_id,
            "bm25",
        )

        dense_rank = find_rank(
            hits,
            gold_doc_id,
            "dense",
        )

        fused_rank = find_rank(
            hits,
            gold_doc_id,
            "fused",
        )

        gold_rows.append(
            {
                "id": query_id,
                "gold_doc_id": gold_doc_id,
                "gold_length": gold_length,
                "median_candidate_length": median_candidate_length,
                "length_ratio": (
                    gold_length / median_candidate_length
                    if median_candidate_length
                    else None
                ),
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank,
                "fused_rank": fused_rank,
            }
        )

    overall_corr = spearman(
        all_lengths,
        all_bm25,
    )

    print(
        f"Overall Spearman(length, BM25): "
        f"{overall_corr:.4f}"
    )

    if query_correlations:
        print(
            f"Mean per-query Spearman       : "
            f"{mean(query_correlations):.4f}"
        )

        print(
            f"Median per-query Spearman     : "
            f"{median(query_correlations):.4f}"
        )

    print("\nGold-document diagnostics")
    print("-" * 90)

    print(
        f"{'ID':<5}"
        f"{'GoldLen':<10}"
        f"{'MedianLen':<12}"
        f"{'Ratio':<10}"
        f"{'BM25':<8}"
        f"{'Dense':<8}"
        f"{'Fused'}"
    )

    for row in gold_rows:
        print(
            f"{row['id']:<5}"
            f"{row['gold_length']:<10}"
            f"{row['median_candidate_length']:<12.1f}"
            f"{row['length_ratio']:<10.2f}"
            f"{row['bm25_rank']:<8}"
            f"{row['dense_rank']:<8}"
            f"{row['fused_rank']}"
        )

    long_gold_bm25_worse = [
        row
        for row in gold_rows
        if (
            row["length_ratio"] is not None
            and row["length_ratio"] > 1.0
            and row["bm25_rank"] is not None
            and row["dense_rank"] is not None
            and row["bm25_rank"] > row["dense_rank"]
        )
    ]

    print("\nLong-gold cases where BM25 ranks worse than dense")
    print("-" * 90)

    for row in long_gold_bm25_worse:
        print(
            f"{row['id']}: "
            f"length_ratio={row['length_ratio']:.2f}, "
            f"BM25={row['bm25_rank']}, "
            f"Dense={row['dense_rank']}, "
            f"Fused={row['fused_rank']}"
        )

    output = {
        "overall_spearman_length_bm25": overall_corr,
        "mean_per_query_spearman": (
            mean(query_correlations)
            if query_correlations
            else None
        ),
        "median_per_query_spearman": (
            median(query_correlations)
            if query_correlations
            else None
        ),
        "gold_documents": gold_rows,
        "long_gold_bm25_worse": long_gold_bm25_worse,
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
        f"\nSaved detailed results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()