from rubbish_rag.judge import (
    HeuristicFeatureJudge,
)


def test_help_query_skips_retrieval():
    judge = HeuristicFeatureJudge()

    result = judge.pre_retrieval(
        "می‌تونی کمکم کنی؟"
    )

    assert (
        result.needs_retrieval
        is False
    )

    assert result.top_k == 0


def test_generic_debt_requires_type():
    judge = HeuristicFeatureJudge()

    result = judge.pre_retrieval(
        "بدهی عقب افتاده را دادم ولی رتبم عوض نشد"
    )

    assert (
        result.needs_clarification
        is True
    )

    assert (
        "debt_type"
        in result.missing_slots
    )


def test_status_only_installment_requires_intent():
    judge = HeuristicFeatureJudge()

    result = judge.pre_retrieval(
        "قسطام عقب افتاده"
    )

    assert (
        result.needs_clarification
        is True
    )

    assert (
        "intent"
        in result.missing_slots
    )


def test_causal_score_query_does_not_overclarify():
    judge = HeuristicFeatureJudge()

    result = judge.pre_retrieval(
        "درآمدم بالاست ولی امتیازم پایینه، چرا؟"
    )

    assert (
        result.needs_clarification
        is False
    )


def test_top_k_is_from_allowed_budget():
    judge = HeuristicFeatureJudge()

    queries = [
        "رتبه C2 یعنی چی؟",
        "چک برگشتی چه اثری دارد؟",
        "بانک صادرات به رتبه E3 وام میده؟",
        "تفاوت رتبه C1 و C2 چیست؟",
    ]

    allowed = {
        0,
        2,
        3,
        5,
        8,
    }

    for query in queries:

        result = (
            judge.pre_retrieval(
                query
            )
        )

        assert (
            result.top_k
            in allowed
        )


def test_high_doc_id_is_not_poison_by_itself():
    judge = HeuristicFeatureJudge()

    hits = [
        {
            "doc_id": 9999,
            "text": (
                "این یک متن عادی درباره "
                "رتبه اعتباری است."
            ),
            "fused": 0.85,
        }
    ]

    result = judge.post_retrieval(
        "رتبه اعتباری چیست؟",
        hits,
    )

    assert (
        result.poison_detected
        is False
    )