import csv
import json
import math
from collections import Counter
from pathlib import Path

from rubbish_rag.normalize_fa import normalize_fa


CORPUS_PATH = Path("corpus/test.csv")
BENCH_PATH = Path("server/visible_bench.json")


def tokenize(text):
    text = normalize_fa(text or "")

    punctuation = "؟?،,.:;!()[]«»\"'"

    return [
        token.strip(punctuation)
        for token in text.split()
        if token.strip(punctuation)
    ]


def cosine_overlap(a_tokens, b_tokens):
    a = Counter(a_tokens)
    b = Counter(b_tokens)

    if not a or not b:
        return 0.0

    dot = sum(
        value * b.get(token, 0)
        for token, value in a.items()
    )

    a_norm = math.sqrt(
        sum(v * v for v in a.values())
    )

    b_norm = math.sqrt(
        sum(v * v for v in b.values())
    )

    if not a_norm or not b_norm:
        return 0.0

    return dot / (a_norm * b_norm)


def load_corpus():
    rows = []

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        for doc_id, row in enumerate(
            csv.DictReader(f)
        ):
            rows.append(
                {
                    "doc_id": doc_id,
                    "question": (
                        row.get("Question")
                        or ""
                    ),
                    "brief": (
                        row.get("BriefAnswer")
                        or ""
                    ),
                    "keyword": (
                        row.get("Keyword")
                        or ""
                    ),
                    "category": (
                        row.get("Category")
                        or ""
                    ),
                }
            )

    return rows


def representation(doc, mode):
    if mode == "question_only":
        return doc["question"]

    if mode == "question_plus":
        return " ".join(
            [
                doc["question"],
                doc["question"],
                doc["keyword"],
                doc["category"],
                doc["brief"],
            ]
        )

    if mode == "no_question":
        return " ".join(
            [
                doc["keyword"],
                doc["category"],
                doc["brief"],
            ]
        )

    raise ValueError(mode)


def rank_docs(query, docs, mode):
    query_tokens = tokenize(query)

    scored = []

    for doc in docs:
        score = cosine_overlap(
            query_tokens,
            tokenize(
                representation(
                    doc,
                    mode,
                )
            ),
        )

        scored.append(
            (
                score,
                doc["doc_id"],
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return scored


def gold_rank(ranked, gold):
    for rank, (_, doc_id) in enumerate(
        ranked,
        start=1,
    ):
        if str(doc_id) == str(gold):
            return rank

    return None


def main():
    docs = load_corpus()

    with BENCH_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        bench = json.load(f)

    modes = [
        "question_plus",
        "question_only",
        "no_question",
    ]

    stats = {
        mode: {
            "h1": 0,
            "h3": 0,
            "h5": 0,
            "h20": 0,
        }
        for mode in modes
    }

    exact_matches = 0

    print(
        "\nLocal Retrieval Leakage / Signal Check"
    )
    print("=" * 105)

    for item in bench:
        gold = int(item["gold_doc_id"])
        query = item["query"]

        gold_doc = docs[gold]

        normalized_query = normalize_fa(
            query
        ).strip()

        normalized_gold_question = (
            normalize_fa(
                gold_doc["question"]
            ).strip()
        )

        exact = (
            normalized_query
            == normalized_gold_question
        )

        exact_matches += int(exact)

        q_similarity = cosine_overlap(
            tokenize(query),
            tokenize(
                gold_doc["question"]
            ),
        )

        ranks = {}

        for mode in modes:
            ranked = rank_docs(
                query,
                docs,
                mode,
            )

            rank = gold_rank(
                ranked,
                gold,
            )

            ranks[mode] = rank

            if rank is not None:
                stats[mode]["h1"] += int(
                    rank <= 1
                )
                stats[mode]["h3"] += int(
                    rank <= 3
                )
                stats[mode]["h5"] += int(
                    rank <= 5
                )
                stats[mode]["h20"] += int(
                    rank <= 20
                )

        print(
            f"{item['id']:<5} "
            f"exactQ={str(exact):<5} "
            f"Qsim={q_similarity:.3f}  "
            f"Q+={str(ranks['question_plus']):<4} "
            f"Q={str(ranks['question_only']):<4} "
            f"NoQ={str(ranks['no_question']):<4}"
        )

    total = len(bench)

    print("\nExact query / corpus-question overlap")
    print("-" * 105)

    print(
        f"Exact normalized matches: "
        f"{exact_matches}/{total} "
        f"({100 * exact_matches / total:.1f}%)"
    )

    print("\nRetrieval ablation")
    print("-" * 105)

    for mode in modes:
        s = stats[mode]

        print(
            f"{mode:<16} "
            f"H@1={s['h1']}/{total} "
            f"({100*s['h1']/total:.1f}%)  "
            f"H@3={s['h3']}/{total} "
            f"({100*s['h3']/total:.1f}%)  "
            f"H@5={s['h5']}/{total} "
            f"({100*s['h5']/total:.1f}%)  "
            f"H@20={s['h20']}/{total} "
            f"({100*s['h20']/total:.1f}%)"
        )


if __name__ == "__main__":
    main()