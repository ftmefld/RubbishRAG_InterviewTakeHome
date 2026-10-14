import csv
import json
from pathlib import Path

from rubbish_rag.normalize_fa import normalize_fa


PARAPHRASE_PATH = Path(
    "analysis/paraphrase_eval.json"
)

CORPUS_PATH = Path(
    "corpus/test.csv"
)


def token_jaccard(a, b):
    a_tokens = set(
        normalize_fa(a).split()
    )

    b_tokens = set(
        normalize_fa(b).split()
    )

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    return (
        len(a_tokens & b_tokens)
        / len(a_tokens | b_tokens)
    )


def main():
    with PARAPHRASE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        cases = json.load(f)

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        corpus = list(
            csv.DictReader(f)
        )

    corpus_questions = []

    for index, row in enumerate(corpus):
        question = (
            row.get("Question")
            or ""
        )

        corpus_questions.append(
            {
                "doc_id": index,
                "question": question,
                "normalized": normalize_fa(
                    question
                ),
            }
        )

    exact_matches = 0

    print(
        "\nParaphrase Leakage Validation"
    )

    print("=" * 100)

    for case in cases:
        query = case["query"]

        normalized_query = normalize_fa(
            query
        )

        exact = [
            row
            for row in corpus_questions
            if (
                row["normalized"]
                == normalized_query
            )
        ]

        if exact:
            exact_matches += 1

        best = max(
            corpus_questions,
            key=lambda row: token_jaccard(
                query,
                row["question"],
            ),
        )

        similarity = token_jaccard(
            query,
            best["question"],
        )

        warning = (
            "  <-- VERY CLOSE"
            if similarity >= 0.85
            else ""
        )

        print(
            f"{case['id']:<4} "
            f"gold={case['gold_doc_id']:<3} "
            f"exact={bool(exact)!s:<5} "
            f"best_doc={best['doc_id']:<3} "
            f"jaccard={similarity:.3f}"
            f"{warning}"
        )

        print(
            f"     Q: {query}"
        )

        print(
            f"     Best corpus question: "
            f"{best['question']}"
        )

    print("\n" + "=" * 100)

    print(
        f"Exact normalized corpus matches: "
        f"{exact_matches}/{len(cases)}"
    )

    if exact_matches == 0:
        print(
            "PASS: no paraphrase is an exact "
            "normalized corpus Question."
        )
    else:
        print(
            "FAIL: replace exact-match "
            "paraphrases before evaluation."
        )


if __name__ == "__main__":
    main()