import json
import statistics
from pathlib import Path


BENCH_PATH = Path("server/visible_bench.json")
TRACE_PATH = Path("traces/trace_baseline_bench.jsonl")


def load_jsonl(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():
    with BENCH_PATH.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)

    traces = load_jsonl(TRACE_PATH)

    # Keep the most recent retrieval trace for each query.
    trace_by_query = {}

    for trace in traces:
        query = trace.get("q")
        hits = trace.get("hits")

        if query and hits:
            trace_by_query[query] = trace

    rows = []

    for example in benchmark:
        query_id = example["id"]
        query = example["query"]
        gold_doc_id = example["gold_doc_id"]

        trace = trace_by_query.get(query)

        if trace is None:
            print(f"[WARNING] No trace found for {query_id}")
            continue

        hits = trace["hits"]
        retrieved_ids = [hit["doc_id"] for hit in hits]

        try:
            gold_rank = retrieved_ids.index(gold_doc_id) + 1
        except ValueError:
            gold_rank = None

        hit_at_3 = gold_doc_id in retrieved_ids[:3]
        hit_at_5 = gold_doc_id in retrieved_ids[:5]

        if len(hits) >= 5:
            top1 = hits[0]["fused"]
            top5 = hits[4]["fused"]
            spread = top1 - top5
        else:
            top1 = hits[0]["fused"] if hits else None
            top5 = None
            spread = None

        poison_top3 = any(
            doc_id >= 9000
            for doc_id in retrieved_ids[:3]
        )

        rows.append(
            {
                "id": query_id,
                "gold_rank": gold_rank,
                "hit3": hit_at_3,
                "hit5": hit_at_5,
                "top1": top1,
                "top5": top5,
                "spread": spread,
                "poison_top3": poison_top3,
            }
        )

    print("\nPer-query retrieval diagnostics")
    print("-" * 88)

    print(
        f"{'ID':<5}"
        f"{'GoldRank':<10}"
        f"{'Hit@3':<8}"
        f"{'Hit@5':<8}"
        f"{'Top1':<10}"
        f"{'Top5':<10}"
        f"{'Spread':<10}"
        f"{'Poison@3'}"
    )

    for row in rows:
        rank = row["gold_rank"] if row["gold_rank"] is not None else "MISS"

        print(
            f"{row['id']:<5}"
            f"{str(rank):<10}"
            f"{str(row['hit3']):<8}"
            f"{str(row['hit5']):<8}"
            f"{row['top1']:<10.4f}"
            f"{row['top5']:<10.4f}"
            f"{row['spread']:<10.4f}"
            f"{row['poison_top3']}"
        )

    n = len(rows)

    hit3 = sum(row["hit3"] for row in rows) / n
    hit5 = sum(row["hit5"] for row in rows) / n

    spreads = [
        row["spread"]
        for row in rows
        if row["spread"] is not None
    ]

    poison_rate = (
        sum(row["poison_top3"] for row in rows) / n
    )

    print("\nAggregate retrieval metrics")
    print("-" * 40)

    print(f"Queries analyzed : {n}")
    print(f"Hit@3            : {hit3:.3f} ({hit3 * 100:.1f}%)")
    print(f"Hit@5            : {hit5:.3f} ({hit5 * 100:.1f}%)")

    print(f"Mean spread      : {statistics.mean(spreads):.4f}")
    print(f"Median spread    : {statistics.median(spreads):.4f}")
    print(f"Min spread       : {min(spreads):.4f}")
    print(f"Max spread       : {max(spreads):.4f}")

    print(f"Poison in Top-3  : {poison_rate:.3f} ({poison_rate * 100:.1f}%)")

    print("\nHit@3 failures")
    print("-" * 40)

    for row in rows:
        if not row["hit3"]:
            print(
                f"{row['id']}: "
                f"gold_rank={row['gold_rank']}, "
                f"spread={row['spread']:.4f}, "
                f"poison_top3={row['poison_top3']}"
            )


if __name__ == "__main__":
    main()