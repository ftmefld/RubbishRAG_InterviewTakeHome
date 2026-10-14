import csv
import json
from collections import Counter
from pathlib import Path

from rubbish_rag.normalize_fa import normalize_fa


PARAPHRASE_PATH = Path(
    "analysis/paraphrase_eval.json"
)

CORPUS_PATH = Path(
    "corpus/test.csv"
)

MISS_IDS = {
    "v3",
    "v9",
    "v10",
    "v12",
}

TOP_SHOW = 10


# ---------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------

def token_counter(text):
    normalized = normalize_fa(
        text or ""
    )

    return Counter(
        normalized.split()
    )


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0

    common = (
        set(a.keys())
        & set(b.keys())
    )

    numerator = sum(
        a[token] * b[token]
        for token in common
    )

    a_norm = sum(
        value * value
        for value in a.values()
    ) ** 0.5

    b_norm = sum(
        value * value
        for value in b.values()
    ) ** 0.5

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return (
        numerator
        / (a_norm * b_norm)
    )


def token_jaccard(a_text, b_text):
    a = set(
        normalize_fa(
            a_text or ""
        ).split()
    )

    b = set(
        normalize_fa(
            b_text or ""
        ).split()
    )

    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 0.0

    return (
        len(a & b)
        / len(union)
    )


# ---------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------

def load_corpus():

    docs = []

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row_index, row in enumerate(
            reader
        ):

            raw_doc_id = row.get(
                "doc_id"
            )

            if raw_doc_id not in (
                None,
                "",
            ):
                try:
                    doc_id = int(
                        raw_doc_id
                    )
                except ValueError:
                    doc_id = (
                        raw_doc_id
                    )
            else:
                doc_id = row_index

            question = (
                row.get("Question")
                or ""
            )

            keyword = (
                row.get("Keyword")
                or ""
            )

            category = (
                row.get("Category")
                or ""
            )

            brief = (
                row.get("BriefAnswer")
                or ""
            )

            answer = (
                row.get("Answer")
                or ""
            )

            docs.append(
                {
                    "doc_id": doc_id,
                    "question": question,
                    "keyword": keyword,
                    "category": category,
                    "brief": brief,
                    "answer": answer,
                    "question_tokens": (
                        token_counter(
                            question
                        )
                    ),
                }
            )

    return docs


# ---------------------------------------------------------------------
# Current Local Question-only ranking
# ---------------------------------------------------------------------

def rank_local_question_only(
    query,
    docs,
):
    q_tokens = token_counter(
        query
    )

    ranked = []

    for doc in docs:

        score = cosine_similarity(
            q_tokens,
            doc[
                "question_tokens"
            ],
        )

        item = dict(doc)

        item[
            "question_score"
        ] = score

        item[
            "jaccard"
        ] = token_jaccard(
            query,
            doc["question"],
        )

        ranked.append(
            item
        )

    ranked.sort(
        key=lambda item: (
            item[
                "question_score"
            ],
            -(
                item["doc_id"]
                if isinstance(
                    item["doc_id"],
                    int,
                )
                else 0
            ),
        ),
        reverse=True,
    )

    return ranked


def find_rank(
    ranked,
    gold_doc_id,
):
    target = str(
        gold_doc_id
    )

    for rank, item in enumerate(
        ranked,
        start=1,
    ):
        if str(
            item["doc_id"]
        ) == target:
            return rank, item

    return None, None


def truncate(text, n=120):
    text = (
        text
        or ""
    ).replace(
        "\n",
        " ",
    )

    if len(text) <= n:
        return text

    return (
        text[: n - 3]
        + "..."
    )


# ---------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------

def main():

    with PARAPHRASE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        cases = json.load(f)

    docs = load_corpus()

    cases = [
        case
        for case in cases
        if case["id"] in MISS_IDS
    ]

    print(
        "\nRetrieval Miss Diagnostic"
    )

    print(
        "=" * 100
    )

    print(
        f"Corpus documents: "
        f"{len(docs)}"
    )

    print(
        "Retriever tested: "
        "current Local Question-only "
        "normalized token cosine"
    )

    print(
        "Remote API calls: 0"
    )

    summary = []

    for case in cases:

        case_id = case["id"]
        query = case["query"]
        gold = case["gold_doc_id"]

        ranked = (
            rank_local_question_only(
                query,
                docs,
            )
        )

        gold_rank, gold_doc = (
            find_rank(
                ranked,
                gold,
            )
        )

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"{case_id} | "
            f"gold={gold}"
        )

        print(
            "-" * 100
        )

        print(
            f"QUERY:"
        )

        print(
            f"  {query}"
        )

        if gold_doc is None:

            print(
                "\nGold document "
                "not found in corpus."
            )

            continue

        print(
            f"\nGOLD LOCAL RANK:"
            f" {gold_rank}"
        )

        print(
            f"GOLD QUESTION SCORE:"
            f" "
            f"{gold_doc['question_score']:.4f}"
        )

        print(
            f"GOLD JACCARD:"
            f" "
            f"{gold_doc['jaccard']:.4f}"
        )

        print(
            "\nGOLD DOCUMENT"
        )

        print(
            f"  Question : "
            f"{gold_doc['question']}"
        )

        print(
            f"  Keyword  : "
            f"{gold_doc['keyword']}"
        )

        print(
            f"  Category : "
            f"{gold_doc['category']}"
        )

        print(
            f"  Brief    : "
            f"{truncate(gold_doc['brief'])}"
        )

        print(
            "\nTOP LOCAL DISTRACTORS"
        )

        for rank, item in enumerate(
            ranked[:TOP_SHOW],
            start=1,
        ):

            marker = (
                " <-- GOLD"
                if str(
                    item["doc_id"]
                ) == str(gold)
                else ""
            )

            print(
                f"\n#{rank:<2} "
                f"doc={item['doc_id']} "
                f"score="
                f"{item['question_score']:.4f} "
                f"jac="
                f"{item['jaccard']:.4f}"
                f"{marker}"
            )

            print(
                f"    Q: "
                f"{truncate(item['question'])}"
            )

            print(
                f"    K: "
                f"{truncate(item['keyword'])}"
            )

            print(
                f"    C: "
                f"{truncate(item['category'])}"
            )

        summary.append(
            {
                "id": case_id,
                "gold": gold,
                "gold_rank": gold_rank,
                "gold_score": (
                    gold_doc[
                        "question_score"
                    ]
                ),
                "gold_jaccard": (
                    gold_doc[
                        "jaccard"
                    ]
                ),
            }
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SUMMARY"
    )

    print(
        "-" * 100
    )

    for row in summary:

        rank = row[
            "gold_rank"
        ]

        if rank <= 20:
            interpretation = (
                "should already be "
                "inside Local Top-20"
            )

        elif rank <= 30:
            interpretation = (
                "near miss; candidate "
                "budget may help"
            )

        elif rank <= 50:
            interpretation = (
                "moderate semantic "
                "ranking failure"
            )

        else:
            interpretation = (
                "strong retrieval "
                "representation failure"
            )

        print(
            f"{row['id']:<4} "
            f"gold={row['gold']:<4} "
            f"rank={rank:<4} "
            f"score="
            f"{row['gold_score']:.4f} "
            f"jac="
            f"{row['gold_jaccard']:.4f} "
            f"| {interpretation}"
        )


if __name__ == "__main__":
    main()