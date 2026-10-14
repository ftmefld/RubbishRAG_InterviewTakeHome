import json
from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline  # register components

from analysis.eval_hybrid_full_pipeline import (
    load_remote_hits_from_traces,
)

from rubbish_rag.pipeline import (
    _hybrid_union,
    _local_question_retrieve,
    _load_local_question_index,
    _rrf_fusion,
    _should_use_rrf,
)


EVAL_PATH = Path(
    "analysis/paraphrase_eval.json"
)

SOURCE_TOPK = 20
SHOW_TOPK = 10


def rank_of(hits, gold_doc_id):
    target = str(
        gold_doc_id
    )

    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        if str(
            hit.get("doc_id")
        ) == target:
            return rank

    return None


def truncate(text, n=130):
    text = (
        text
        or ""
    ).replace(
        "\n",
        " ",
    )

    if len(text) <= n:
        return text

    return text[: n - 3] + "..."


def build_doc_map():
    return {
        str(doc["doc_id"]): doc
        for doc in _load_local_question_index()
    }


def main():

    with EVAL_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        cases = json.load(f)

    queries = [
        case["query"]
        for case in cases
    ]

    remote_cache = (
        load_remote_hits_from_traces(
            queries
        )
    )

    reranker = R.get(
        "reranker"
    )

    corpus_map = (
        build_doc_map()
    )

    print(
        "\nTop-5 Near-Miss Diagnostic"
    )

    print(
        "=" * 120
    )

    print(
        "Goal: inspect gold documents that are "
        "retrieved but remain below final Top-5."
    )

    print(
        "Remote API calls: 0"
    )

    near_misses = []

    for case in cases:

        query = case["query"]
        gold = case["gold_doc_id"]

        remote_hits = [
            dict(hit)
            for hit in remote_cache[
                query
            ]
        ]

        if (
            remote_hits
            and _should_use_rrf(
                remote_hits
            )
        ):
            remote_hits = (
                _rrf_fusion(
                    remote_hits
                )
            )

        local_hits = (
            _local_question_retrieve(
                query,
                topk=SOURCE_TOPK,
            )
        )

        candidates = _hybrid_union(
            remote_hits,
            local_hits,
        )

        candidate_rank = rank_of(
            candidates,
            gold,
        )

        # Pure retrieval miss -> not a reranking problem.
        if candidate_rank is None:
            continue

        ranked = reranker(
            query,
            [
                dict(hit)
                for hit in candidates
            ],
        )

        final_rank = rank_of(
            ranked,
            gold,
        )

        # We only care about golds outside Top-5.
        if (
            final_rank is None
            or final_rank <= 5
        ):
            continue

        near_misses.append(
            {
                "id": case["id"],
                "query": query,
                "gold": gold,
                "candidate_rank": (
                    candidate_rank
                ),
                "final_rank": (
                    final_rank
                ),
            }
        )

        print(
            "\n"
            + "=" * 120
        )

        print(
            f"{case['id']} | "
            f"gold={gold} | "
            f"candidate="
            f"{candidate_rank} | "
            f"reranked="
            f"{final_rank}"
        )

        print(
            "-" * 120
        )

        print(
            f"QUERY:\n  {query}"
        )

        gold_meta = corpus_map.get(
            str(gold),
            {},
        )

        print(
            "\nGOLD CORPUS QUESTION:"
        )

        print(
            " ",
            gold_meta.get(
                "question",
                "<not found>",
            ),
        )

        print(
            "\nFINAL TOP CANDIDATES:"
        )

        for rank, hit in enumerate(
            ranked[:SHOW_TOPK],
            start=1,
        ):

            doc_id = hit.get(
                "doc_id"
            )

            meta = corpus_map.get(
                str(doc_id),
                {},
            )

            marker = (
                "  <-- GOLD"
                if str(doc_id)
                == str(gold)
                else ""
            )

            print(
                f"\n#{rank:<2} "
                f"doc={doc_id} "
                f"source="
                f"{str(hit.get('retrieval_source')):<12} "
                f"prior="
                f"{hit.get('retrieval_prior', 0.0):.5f} "
                f"local="
                f"{hit.get('local_score', 0.0):.3f} "
                f"lex="
                f"{hit.get('rerank_score', 0):<2} "
                f"verified="
                f"{hit.get('verified_corpus')} "
                f"polarity="
                f"{hit.get('polarity_compatible')}"
                f"{marker}"
            )

            print(
                "    Q:",
                truncate(
                    meta.get(
                        "question",
                        "",
                    )
                ),
            )

            print(
                "    K:",
                truncate(
                    meta.get(
                        "keyword",
                        "",
                    )
                ),
            )

            print(
                "    C:",
                truncate(
                    meta.get(
                        "category",
                        "",
                    )
                ),
            )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "SUMMARY"
    )

    print(
        "-" * 120
    )

    if not near_misses:
        print(
            "No retrieved gold documents "
            "remain below Top-5."
        )

    else:
        for row in near_misses:
            print(
                f"{row['id']:<4} "
                f"gold={row['gold']:<4} "
                f"candidate="
                f"{row['candidate_rank']:<3} "
                f"reranked="
                f"{row['final_rank']:<3}"
            )


if __name__ == "__main__":
    main()