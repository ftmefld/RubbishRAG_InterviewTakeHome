import json
from pathlib import Path

import rubbish_rag as R

# Import pipeline so all decorators are registered.
import rubbish_rag.pipeline  # noqa: F401

from rubbish_rag.pipeline import (
    _hybrid_union,
    _local_question_retrieve,
    _rrf_fusion,
    _should_use_rrf,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PARAPHRASE_PATH = Path(
    "analysis/paraphrase_eval.json"
)

TRACE_PATH = Path(
    "traces/trace.jsonl"
)

OUTPUT_PATH = Path(
    "analysis/hybrid_full_pipeline_results.json"
)

SOURCE_TOPK = 20


# ---------------------------------------------------------------------
# Trace replay
# ---------------------------------------------------------------------

def _extract_hits(record):
    """
    Extract retrieval hits from a trace record.

    Different trace versions may place hits under slightly different
    keys, so we support a few known layouts.
    """

    for key in (
        "hits",
        "results",
        "candidates",
    ):
        value = record.get(key)

        if (
            isinstance(value, list)
            and (
                not value
                or isinstance(
                    value[0],
                    dict,
                )
            )
        ):
            return value

    for parent_key in (
        "response",
        "result",
        "data",
    ):
        parent = record.get(
            parent_key
        )

        if not isinstance(
            parent,
            dict,
        ):
            continue

        for key in (
            "hits",
            "results",
            "candidates",
        ):
            value = parent.get(key)

            if (
                isinstance(value, list)
                and (
                    not value
                    or isinstance(
                        value[0],
                        dict,
                    )
                )
            ):
                return value

    return None


def load_remote_hits_from_traces(
    queries,
):
    """
    Recover the most recent Top-20 remote retrieval result
    for every paraphrase query.

    This avoids consuming additional /retrieve quota.
    """

    wanted = set(
        queries
    )

    found = {}
    latest_records = {}

    if not TRACE_PATH.exists():
        raise RuntimeError(
            f"Trace file not found: "
            f"{TRACE_PATH}"
        )

    with TRACE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError:
                continue

            query = record.get("q")

            if query not in wanted:
                continue

            latest_records[
                query
            ] = record

            try:
                trace_topk = int(
                    record.get(
                        "topk",
                        0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                trace_topk = 0

            if trace_topk < SOURCE_TOPK:
                continue

            hits = _extract_hits(
                record
            )

            if hits is None:
                continue

            # Keep the latest valid Top-20 record.
            found[query] = [
                dict(hit)
                for hit in hits
            ]

    missing = [
        query
        for query in queries
        if query not in found
    ]

    if missing:
        print(
            "\nCould not recover Top-20 "
            "remote hits for:"
        )

        for query in missing:
            print(
                f"  - {query}"
            )

            record = (
                latest_records.get(
                    query
                )
            )

            if record is not None:
                print(
                    "    Latest trace keys:",
                    sorted(
                        record.keys()
                    ),
                )

        raise RuntimeError(
            "Some remote Top-20 results "
            "could not be recovered from "
            "traces/trace.jsonl. "
            "No API calls were made."
        )

    return found


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def rank_of_hits(
    hits,
    gold_doc_id,
):
    gold = str(
        gold_doc_id
    )

    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        if str(
            hit.get("doc_id")
        ) == gold:
            return rank

    return None


def compute_metrics(
    ranks,
):
    total = len(
        ranks
    )

    def hit_at(k):
        if not total:
            return 0.0

        count = sum(
            1
            for rank in ranks
            if (
                rank is not None
                and rank <= k
            )
        )

        return count / total

    if total:
        mrr = sum(
            (
                1.0 / rank
                if rank is not None
                else 0.0
            )
            for rank in ranks
        ) / total
    else:
        mrr = 0.0

    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "hit_at_20": hit_at(20),
        "mrr": mrr,
    }


def print_metrics(
    name,
    metrics,
):
    print(
        f"\n{name}"
    )

    print(
        "-" * 65
    )

    print(
        f"Hit@1  : "
        f"{100 * metrics['hit_at_1']:.1f}%"
    )

    print(
        f"Hit@3  : "
        f"{100 * metrics['hit_at_3']:.1f}%"
    )

    print(
        f"Hit@5  : "
        f"{100 * metrics['hit_at_5']:.1f}%"
    )

    print(
        f"Hit@20 : "
        f"{100 * metrics['hit_at_20']:.1f}%"
    )

    print(
        f"MRR     : "
        f"{metrics['mrr']:.3f}"
    )


def show_rank(rank):
    if rank is None:
        return "-"

    return str(
        rank
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def main():

    # -------------------------------------------------------------
    # Load synthetic non-exact paraphrases
    # -------------------------------------------------------------

    with PARAPHRASE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        cases = json.load(f)

    queries = [
        case["query"]
        for case in cases
    ]

    # -------------------------------------------------------------
    # Replay previously executed Remote Top-20 calls
    # -------------------------------------------------------------

    remote_cache = (
        load_remote_hits_from_traces(
            queries
        )
    )

    reranker = R.get(
        "reranker"
    )

    resolver = R.get(
        "resolver"
    )

    print(
        "\nHybrid Full-Pipeline "
        "Paraphrase Evaluation"
    )

    print(
        "=" * 90
    )

    print(
        f"Queries: {len(cases)}"
    )

    print(
        "Remote source: replayed "
        "from traces/trace.jsonl"
    )

    print(
        "New /retrieve calls: 0"
    )

    print(
        f"Remote candidates/query : "
        f"{SOURCE_TOPK}"
    )

    print(
        f"Local candidates/query  : "
        f"{SOURCE_TOPK}"
    )

    print(
        "Hybrid pool             : "
        "full deduplicated union"
    )

    # -------------------------------------------------------------
    # Evaluation storage
    # -------------------------------------------------------------

    candidate_ranks = []
    reranked_ranks = []

    union_sizes = []

    answer_count = 0
    clarify_count = 0

    citation_hits = 0

    details = []

    # -------------------------------------------------------------
    # Run every paraphrase
    # -------------------------------------------------------------

    for case in cases:

        case_id = case[
            "id"
        ]

        query = case[
            "query"
        ]

        gold = case[
            "gold_doc_id"
        ]

        # =========================================================
        # Stage 1A — Remote Top-20
        # =========================================================

        remote_hits = [
            dict(hit)
            for hit in remote_cache[
                query
            ]
        ]

        # Preserve our previous Task B gate.
        if (
            remote_hits
            and _should_use_rrf(
                remote_hits
            )
        ):
            remote_hits = _rrf_fusion(
                remote_hits
            )

        # =========================================================
        # Stage 1B — Local Question Top-20
        # =========================================================

        local_hits = (
            _local_question_retrieve(
                query,
                topk=SOURCE_TOPK,
            )
        )

        # =========================================================
        # Stage 1C — Full deduplicated union
        # =========================================================

        candidates = _hybrid_union(
            remote_hits,
            local_hits,
        )

        union_sizes.append(
            len(candidates)
        )

        # IMPORTANT:
        #
        # We intentionally do NOT truncate candidates to 20 here.
        #
        # Remote Top-20 + Local Top-20 may create roughly
        # 25–40 unique documents.
        #
        # The reranker should decide which candidates deserve
        # the final Top-5.

        candidate_rank = rank_of_hits(
            candidates,
            gold,
        )

        candidate_ranks.append(
            candidate_rank
        )

        # =========================================================
        # Stage 2 — Current reranker
        #
        # Task C:
        # normalized lexical matching
        #
        # Track F:
        # trusted-corpus grounding
        #
        # Track G:
        # polarity compatibility
        #
        # Track E:
        # document diversity
        # =========================================================

        ranked = reranker(
            query,
            [
                dict(hit)
                for hit in candidates
            ],
        )

        reranked_rank = (
            rank_of_hits(
                ranked,
                gold,
            )
        )

        reranked_ranks.append(
            reranked_rank
        )

        # =========================================================
        # Stage 3 — Final Top-5 -> resolver
        # =========================================================

        final_hits = (
            ranked[:5]
        )

        result = resolver(
            query,
            final_hits,
        )

        status = result.get(
            "status"
        )

        cites = result.get(
            "cites",
            [],
        )

        if status == "answer":
            answer_count += 1

        elif status == "clarify":
            clarify_count += 1

        cited_gold = any(
            str(doc_id)
            == str(gold)
            for doc_id in cites
        )

        citation_hits += int(
            cited_gold
        )

        # =========================================================
        # Determine which retrieval source supplied the gold
        # =========================================================

        gold_source = None

        for hit in candidates:

            if str(
                hit.get(
                    "doc_id"
                )
            ) != str(gold):
                continue

            gold_source = (
                hit.get(
                    "retrieval_source"
                )
            )

            break

        # =========================================================
        # Save detailed case result
        # =========================================================

        details.append(
            {
                "id": case_id,
                "query": query,
                "gold_doc_id": gold,

                "remote_count": len(
                    remote_hits
                ),

                "local_count": len(
                    local_hits
                ),

                "union_count": len(
                    candidates
                ),

                "candidate_rank": (
                    candidate_rank
                ),

                "reranked_rank": (
                    reranked_rank
                ),

                "gold_source": (
                    gold_source
                ),

                "status": status,

                "cites": cites,

                "gold_cited": (
                    cited_gold
                ),

                "candidate_doc_ids": [
                    hit.get(
                        "doc_id"
                    )
                    for hit in candidates
                ],

                "reranked_top5": [
                    hit.get(
                        "doc_id"
                    )
                    for hit in final_hits
                ],
            }
        )

        print(
            f"{case_id:<4} "
            f"gold={gold:<3} "
            f"remote={len(remote_hits):<2} "
            f"local={len(local_hits):<2} "
            f"union={len(candidates):<2} "
            f"candidate="
            f"{show_rank(candidate_rank):<3} "
            f"reranked="
            f"{show_rank(reranked_rank):<3} "
            f"source="
            f"{str(gold_source):<12} "
            f"cited="
            f"{cited_gold}"
        )

    # -----------------------------------------------------------------
    # Candidate-pool coverage
    # -----------------------------------------------------------------

    total = len(
        cases
    )

    candidate_found = sum(
        1
        for rank in candidate_ranks
        if rank is not None
    )

    candidate_recall = (
        candidate_found / total
        if total
        else 0.0
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "\nHYBRID FULL-POOL COVERAGE"
    )

    print(
        "-" * 65
    )

    print(
        f"Gold found in candidate pool : "
        f"{candidate_found}/{total} "
        f"({100 * candidate_recall:.1f}%)"
    )

    if union_sizes:
        print(
            f"Average union size           : "
            f"{sum(union_sizes) / len(union_sizes):.1f}"
        )

        print(
            f"Minimum union size           : "
            f"{min(union_sizes)}"
        )

        print(
            f"Maximum union size           : "
            f"{max(union_sizes)}"
        )

    # -----------------------------------------------------------------
    # Candidate ordering metrics
    # -----------------------------------------------------------------

    candidate_metrics = (
        compute_metrics(
            candidate_ranks
        )
    )

    reranked_metrics = (
        compute_metrics(
            reranked_ranks
        )
    )

    print_metrics(
        "HYBRID CANDIDATE ORDER "
        "(before reranking)",
        candidate_metrics,
    )

    print_metrics(
        "AFTER CURRENT RERANKER",
        reranked_metrics,
    )

    # -----------------------------------------------------------------
    # Resolver / citation metrics
    # -----------------------------------------------------------------

    citation_rate = (
        citation_hits / total
        if total
        else 0.0
    )

    print(
        "\nRESOLVER / CITATION"
    )

    print(
        "-" * 65
    )

    print(
        f"Answer status    : "
        f"{answer_count}/{total}"
    )

    print(
        f"Clarify status   : "
        f"{clarify_count}/{total}"
    )

    print(
        f"Gold cited Top-3 : "
        f"{citation_hits}/{total} "
        f"({100 * citation_rate:.1f}%)"
    )

    # -----------------------------------------------------------------
    # Gold-source complementarity
    # -----------------------------------------------------------------

    source_counts = {
        "remote": 0,
        "local": 0,
        "remote+local": 0,
        "missing": 0,
    }

    for row in details:

        source = row[
            "gold_source"
        ]

        if source is None:
            source_counts[
                "missing"
            ] += 1

        elif source in source_counts:
            source_counts[
                source
            ] += 1

    print(
        "\nGOLD SOURCE IN HYBRID POOL"
    )

    print(
        "-" * 65
    )

    print(
        f"Remote only       : "
        f"{source_counts['remote']}"
    )

    print(
        f"Local only        : "
        f"{source_counts['local']}"
    )

    print(
        f"Remote + Local    : "
        f"{source_counts['remote+local']}"
    )

    print(
        f"Missing from both : "
        f"{source_counts['missing']}"
    )

    # -----------------------------------------------------------------
    # Compare to earlier retrieval-only experiment
    # -----------------------------------------------------------------

    previous_path = Path(
        "analysis/"
        "paraphrase_retrieval_results.json"
    )

    previous = None

    if previous_path.exists():

        with previous_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            previous = json.load(f)

        print(
            "\nEARLIER RETRIEVAL-ONLY "
            "REFERENCE"
        )

        print(
            "-" * 65
        )

        for name in (
            "remote",
            "local",
            "hybrid",
        ):
            metrics = previous[
                name
            ]

            print(
                f"{name.upper():<8} "
                f"H@1="
                f"{100 * metrics['hit_at_1']:.1f}% "
                f"H@3="
                f"{100 * metrics['hit_at_3']:.1f}% "
                f"H@5="
                f"{100 * metrics['hit_at_5']:.1f}% "
                f"H@20="
                f"{100 * metrics['hit_at_20']:.1f}% "
                f"MRR="
                f"{metrics['mrr']:.3f}"
            )

    # -----------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "evaluation": (
                    "hybrid_full_pipeline_"
                    "synthetic_non_exact_"
                    "paraphrase_stress_test"
                ),

                "new_remote_calls": 0,

                "remote_topk": (
                    SOURCE_TOPK
                ),

                "local_topk": (
                    SOURCE_TOPK
                ),

                "candidate_pool": (
                    "full_deduplicated_union"
                ),

                "candidate_coverage": {
                    "found": (
                        candidate_found
                    ),
                    "total": total,
                    "recall": (
                        candidate_recall
                    ),
                },

                "union_size": {
                    "mean": (
                        sum(union_sizes)
                        / len(union_sizes)
                        if union_sizes
                        else 0.0
                    ),
                    "min": (
                        min(union_sizes)
                        if union_sizes
                        else 0
                    ),
                    "max": (
                        max(union_sizes)
                        if union_sizes
                        else 0
                    ),
                },

                "candidate_metrics": (
                    candidate_metrics
                ),

                "reranked_metrics": (
                    reranked_metrics
                ),

                "resolver": {
                    "answers": (
                        answer_count
                    ),
                    "clarifications": (
                        clarify_count
                    ),
                    "gold_cited": (
                        citation_hits
                    ),
                    "gold_citation_rate": (
                        citation_rate
                    ),
                },

                "gold_sources": (
                    source_counts
                ),

                "details": details,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        "\nNo new remote retrieval "
        "calls were used."
    )


if __name__ == "__main__":
    main()