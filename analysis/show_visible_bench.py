import json
from pathlib import Path


BENCH_PATH = Path(
    "server/visible_bench.json"
)


def main():
    with BENCH_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        bench = json.load(f)

    for item in bench:
        print("=" * 90)

        print(
            f"{item['id']} | "
            f"gold={item['gold_doc_id']}"
        )

        print(
            item["query"]
        )


if __name__ == "__main__":
    main()