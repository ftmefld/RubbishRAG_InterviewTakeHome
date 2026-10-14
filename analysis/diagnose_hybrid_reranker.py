import json
from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline  # noqa: F401

from analysis.eval_hybrid_full_pipeline import (
    load_remote_hits_from_traces,
)

from rubbish_rag.pipeline import (
    _hybrid_union,
    _local_question_retrieve,
    _rrf_fusion,
    _should_use_rrf,
)


PARAPHRASE_PATH = Path(
    "analysis/paraphrase_eval.json"
)

SOURCE_TOPK = 20


def find_doc(hits, doc_id):
    target = str(doc_id)

    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        if str(
            hit.get("doc_id")
        ) == target:
            return rank, hit

    return None, None


def main():

    with PARAPHRASE_PATH.open(
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

    print(
        "\nHybrid Reranker Diagnostic"
    )

    print(
        "=" * 120
    )

    source_stats = {}

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

        pre_rank, _ = find_doc(
            candidates,
            gold,
        )

        ranked = reranker(
            query,
            [
                dict(hit)
                for hit in candidates
            ],
        )

        post_rank, gold_hit = (
            find_doc(
                ranked,
                gold,
            )
        )

        if gold_hit is None:

            print(
                f"{case['id']:<4} "
                f"gold={gold:<3} "
                f"MISSING"
            )

            continue

        source = gold_hit.get(
            "retrieval_source"
        )

        verified = gold_hit.get(
            "verified_corpus"
        )

        polarity = gold_hit.get(
            "polarity_compatible"
        )

        prior = gold_hit.get(
            "retrieval_prior",
            0.0,
        )

        local_score = gold_hit.get(
            "local_score",
            0.0,
        )

        lexical = gold_hit.get(
            "rerank_score",
            0,
        )

        fused = gold_hit.get(
            "fused",
            0.0,
        )

        if (
            pre_rank is not None
            and post_rank is not None
        ):
            delta = (
                post_rank
                - pre_rank
            )
        else:
            delta = None

        stats = source_stats.setdefault(
            source,
            {
                "count": 0,
                "verified": 0,
                "improved": 0,
                "same": 0,
                "worse": 0,
            },
        )

        stats["count"] += 1

        if verified:
            stats[
                "verified"
            ] += 1

        if delta is not None:

            if delta < 0:
                stats[
                    "improved"
                ] += 1

            elif delta > 0:
                stats[
                    "worse"
                ] += 1

            else:
                stats[
                    "same"
                ] += 1

        print(
            f"{case['id']:<4} "
            f"gold={gold:<3} "
            f"source={str(source):<12} "
            f"rank={str(pre_rank):>2}"
            f"->{str(post_rank):<2} "
            f"verified={str(verified):<5} "
            f"polarity={str(polarity):<5} "
            f"prior={prior:.5f} "
            f"local={local_score:.3f} "
            f"lexical={lexical:<2} "
            f"fused={fused:.3f}"
        )

    print(
        "\n" + "=" * 120
    )

    print(
        "SUMMARY BY GOLD SOURCE"
    )

    print(
        "-" * 120
    )

    for source, stats in (
        source_stats.items()
    ):

        print(
            f"{str(source):<14} "
            f"n={stats['count']:<2} "
            f"verified="
            f"{stats['verified']}/"
            f"{stats['count']} "
            f"improved="
            f"{stats['improved']} "
            f"same="
            f"{stats['same']} "
            f"worse="
            f"{stats['worse']}"
        )


if __name__ == "__main__":
    main()