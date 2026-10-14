"""Thin end-to-end orchestrator for the final RubbishRAG pipeline.

The orchestrator intentionally does not reimplement normalization, slot
extraction, retrieval, reranking, grounding, clarification, or poison
detection. Those responsibilities remain in their canonical modules:

    normalize_fa.py
    slots.py
    pipeline.py
    judge.py

Its job is only to coordinate those components into one coherent adaptive
RAG state machine.
"""

from typing import Any, Dict, List, Optional

import rubbish_rag as R

# Importing pipeline registers the final chunker/retriever/reranker/resolver
# through the decorators in rubbish_rag.__init__.
from . import pipeline as _pipeline  # noqa: F401
from .judge import (
    HeuristicFeatureJudge,
    RAGDecisionJudge,
    adaptive_select_hits,
)
from .normalize_fa import normalize_fa
from .slots import extract_slots


MAX_ADAPTIVE_CONTEXT = 8
MIN_EVIDENCE_CONTEXT = 2


def _get_stage(kind: str):
    """Return one registered production stage or fail loudly."""

    stage = R.get(kind)

    if stage is None:
        raise RuntimeError(
            f"RubbishRAG stage '{kind}' is not registered."
        )

    return stage


def _conversational_answer(query: str) -> str:
    """Small non-RAG response for meta/conversational prompts."""

    normalized = normalize_fa(query)

    if any(
        token in normalized
        for token in (
            "سلام",
            "درود",
            "صبح بخیر",
            "شب بخیر",
            "وقت بخیر",
        )
    ):
        return (
            "سلام! می‌توانم درباره اعتبارسنجی، رتبه اعتباری، "
            "وام و تسهیلات، چک و بدهی کمک کنم."
        )

    if (
        "چه سوال" in normalized
        or "چه جور سوال" in normalized
        or "چه چیز" in normalized
    ):
        return (
            "می‌توانید درباره رتبه و امتیاز اعتباری، وام و "
            "تسهیلات، چک صیادی، بدهی و سوابق مالیاتی سؤال کنید."
        )

    return (
        "می‌توانم درباره موضوعات اعتبارسنجی و سوابق اعتباری "
        "راهنمایی کنم."
    )


def _clarification_payload(
    missing_slots: List[str],
    reason_codes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Turn the judge's structured missing slots into a user-facing question."""

    missing = set(
        missing_slots or []
    )

    reasons = set(
        reason_codes or []
    )

    if "debt_type" in missing:
        return {
            "status": "clarify",
            "question": "منظورتان کدام نوع بدهی است؟",
            "options": [
                "بدهی تسهیلاتی / وام",
                "بدهی مالیاتی",
                "چک برگشتی",
                "محکومیت مالی",
                "نمی‌دانم",
            ],
        }

    if "bank" in missing:
        return {
            "status": "clarify",
            "question": (
                "برای کدام بانک درباره دریافت وام یا تسهیلات "
                "سؤال دارید؟"
            ),
            "options": [
                "ملی",
                "صادرات",
                "تجارت",
                "ملت",
                "سپه",
                "پاسارگاد",
                "توسعه صادرات",
                "نمی‌دانم",
            ],
        }

    if (
        "rank" in missing
        or "persona" in missing
    ):
        return {
            "status": "clarify",
            "question": (
                "برای بررسی تغییر رتبه، لطفاً رتبه فعلی یا نوع "
                "شخص (حقیقی/حقوقی) و زمینه مرتبط را مشخص کنید."
            ),
            "options": [
                "شخص حقیقی",
                "شخص حقوقی",
                "رتبه فعلی را می‌دانم",
                "رتبه فعلی را نمی‌دانم",
            ],
        }

    if (
        "repayment_status" in missing
        or "time_since_loan" in missing
    ):
        return {
            "status": "clarify",
            "question": (
                "وضعیت بازپرداخت وام و مدت زمانی که از دریافت یا "
                "پرداخت آن گذشته را مشخص می‌کنید؟"
            ),
            "options": [
                "اقساط منظم پرداخت شده",
                "قسط معوق دارم",
                "وام تازه دریافت شده",
                "وام تسویه شده",
            ],
        }

    if "company_age" in missing:
        return {
            "status": "clarify",
            "question": (
                "شرکت چه مدت است تأسیس شده و آیا سابقه مالی یا "
                "اعتباری قبلی دارد؟"
            ),
            "options": [
                "کمتر از یک سال",
                "بیشتر از یک سال",
                "سابقه اعتباری دارد",
                "سابقه اعتباری ندارد",
            ],
        }

    if "tax_stage" in missing:
        return {
            "status": "clarify",
            "question": (
                "منظورتان از وضعیت مالیات چیست؛ بدهی ثبت‌شده، "
                "جریمه، یا صرفاً پرداخت‌نشدن مالیات؟"
            ),
            "options": [
                "بدهی مالیاتی ثبت شده",
                "جریمه مالیاتی",
                "مالیات پرداخت نشده",
                "مطمئن نیستم",
            ],
        }

    if (
        "intent" in missing
        or "MISSING_USER_INTENT" in reasons
        or "MISSING_GUARANTEE_INTENT" in reasons
    ):
        return {
            "status": "clarify",
            "question": (
                "دقیقاً می‌خواهید بدانید این وضعیت چه اثری روی "
                "رتبه اعتباری، دریافت وام، یا گزارش اعتباری دارد؟"
            ),
            "options": [
                "اثر روی رتبه اعتباری",
                "اثر روی دریافت وام",
                "اثر روی گزارش اعتباری",
                "راه اصلاح وضعیت",
            ],
        }

    return {
        "status": "clarify",
        "question": (
            "برای پاسخ دقیق‌تر، لطفاً کمی اطلاعات بیشتری درباره "
            "وضعیت یا هدفتان بفرمایید."
        ),
        "options": [],
    }


class RubbishRAGOrchestrator:
    """Coordinate the final adaptive RAG components without duplicating logic."""

    def __init__(
        self,
        api_key: str = "demo-key",
        judge: Optional[
            RAGDecisionJudge
        ] = None,
    ):
        # Kept only for backwards compatibility with the starter interface.
        # The registered production retriever owns its own remote-retriever
        # contract and does not receive this key from the orchestrator.
        self.api_key = api_key

        self.judge = (
            judge
            or HeuristicFeatureJudge()
        )

    def run(
        self,
        query: str,
        topk: int = 5,
        conversation: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        """Run the final adaptive end-to-end RAG state machine."""

        query = str(
            query or ""
        ).strip()

        if not query:
            return {
                "status": "clarify",
                "question": "سؤال خود را وارد کنید.",
                "options": [],
            }

        normalized_query = normalize_fa(
            query
        )

        slots = extract_slots(
            query
        )

        # ---------------------------------------------------------
        # 1. System-1 pre-retrieval decision
        # ---------------------------------------------------------

        pre = self.judge.pre_retrieval(
            query,
            conversation=conversation,
            slots=slots,
        )

        if not pre.needs_retrieval:
            return {
                "status": "answer",
                "brief": _conversational_answer(
                    query
                ),
                "cites": [],
                "slots": slots,
                "normalized_query": (
                    normalized_query
                ),
                "orchestration": {
                    "pre": pre.to_dict(),
                    "post": None,
                    "retrieved_candidates": 0,
                    "selected_context": 0,
                },
            }

        if pre.needs_clarification:
            response = (
                _clarification_payload(
                    pre.missing_slots,
                    pre.reason_codes,
                )
            )

            response.update(
                {
                    "slots": slots,
                    "normalized_query": (
                        normalized_query
                    ),
                    "orchestration": {
                        "pre": pre.to_dict(),
                        "post": None,
                        "retrieved_candidates": 0,
                        "selected_context": 0,
                    },
                }
            )

            return response

        # ---------------------------------------------------------
        # 2. Retrieve a broad candidate pool once.
        #
        # Context budgeting controls how many candidates become
        # evidence, not how broad initial candidate discovery is.
        # ---------------------------------------------------------

        retriever = _get_stage(
            "retriever"
        )

        reranker = _get_stage(
            "reranker"
        )

        resolver = _get_stage(
            "resolver"
        )

        legacy_topk = max(
            1,
            min(
                int(topk),
                20,
            ),
        )

        candidate_pool_k = max(
            MAX_ADAPTIVE_CONTEXT,
            legacy_topk,
            pre.top_k,
        )

        raw_hits = retriever(
            pre.query_rewrite
            or query,
            topk=candidate_pool_k,
        )

        ranked_hits = reranker(
            query,
            raw_hits,
        )

        # ---------------------------------------------------------
        # 3. Adaptive evidence selection.
        # ---------------------------------------------------------

        initial_budget = min(
            max(
                pre.top_k,
                MIN_EVIDENCE_CONTEXT,
            ),
            MAX_ADAPTIVE_CONTEXT,
        )

        selected_hits, post = (
            adaptive_select_hits(
                query,
                ranked_hits,
                self.judge,
                max_k=initial_budget,
                min_k=min(
                    MIN_EVIDENCE_CONTEXT,
                    initial_budget,
                ),
            )
        )

        # One bounded expansion using the already-retrieved pool.
        # No second remote retrieval call is made.
        if (
            post.next_action
            == "retrieve_more"
            and initial_budget
            < MAX_ADAPTIVE_CONTEXT
            and len(
                ranked_hits
            )
            > len(
                selected_hits
            )
        ):
            selected_hits, post = (
                adaptive_select_hits(
                    query,
                    ranked_hits,
                    self.judge,
                    max_k=(
                        MAX_ADAPTIVE_CONTEXT
                    ),
                    min_k=min(
                        MIN_EVIDENCE_CONTEXT,
                        MAX_ADAPTIVE_CONTEXT,
                    ),
                )
            )

        orchestration_meta = {
            "pre": pre.to_dict(),
            "post": post.to_dict(),
            "retrieved_candidates": len(
                raw_hits
            ),
            "reranked_candidates": len(
                ranked_hits
            ),
            "selected_context": len(
                selected_hits
            ),
            "initial_context_budget": (
                initial_budget
            ),
            "max_context_budget": (
                MAX_ADAPTIVE_CONTEXT
            ),
        }

        # ---------------------------------------------------------
        # 4. Judge-controlled next action
        # ---------------------------------------------------------

        if post.next_action == "clarify":
            response = (
                _clarification_payload(
                    pre.missing_slots,
                    pre.reason_codes,
                )
            )

            response.update(
                {
                    "slots": slots,
                    "normalized_query": (
                        normalized_query
                    ),
                    "orchestration": (
                        orchestration_meta
                    ),
                }
            )

            return response

        if (
            post.next_action
            == "abstain"
        ):
            return {
                "status": "abstain",
                "brief": (
                    "شواهد بازیابی‌شده برای پاسخ قابل اتکا نیستند؛ "
                    "بهتر است بدون حدس پاسخ داده نشود."
                ),
                "cites": [],
                "slots": slots,
                "normalized_query": (
                    normalized_query
                ),
                "orchestration": (
                    orchestration_meta
                ),
            }

        if (
            post.next_action
            == "retrieve_more"
        ):
            return {
                "status": "abstain",
                "brief": (
                    "شواهد کافی برای پاسخ مطمئن پیدا نشد."
                ),
                "cites": [
                    hit.get(
                        "doc_id"
                    )
                    for hit
                    in selected_hits
                    if hit.get(
                        "doc_id"
                    )
                    is not None
                ][:3],
                "slots": slots,
                "normalized_query": (
                    normalized_query
                ),
                "orchestration": (
                    orchestration_meta
                ),
            }

        # ---------------------------------------------------------
        # 5. Final resolver.
        # ---------------------------------------------------------

        resolved = resolver(
            query,
            selected_hits,
        )

        # Resolver may still discover a domain-specific clarification.
        if (
            resolved.get(
                "status"
            )
            == "clarify"
        ):
            resolved[
                "orchestration"
            ] = orchestration_meta

            resolved[
                "slots"
            ] = slots

            resolved[
                "normalized_query"
            ] = normalized_query

            return resolved

        resolved[
            "orchestration"
        ] = orchestration_meta

        resolved[
            "slots"
        ] = slots

        resolved[
            "normalized_query"
        ] = normalized_query

        return resolved

    def invoke(
        self,
        input_data: Any,
    ) -> Dict[str, Any]:
        """LCEL-style invocation compatible with the starter interface."""

        if isinstance(
            input_data,
            dict,
        ):
            query = (
                input_data.get(
                    "query"
                )
                or input_data.get(
                    "question"
                )
                or ""
            )

            topk = input_data.get(
                "topk",
                5,
            )

            conversation = (
                input_data.get(
                    "conversation"
                )
            )

        else:
            query = str(
                input_data
            )

            topk = 5

            conversation = None

        return self.run(
            query=query,
            topk=topk,
            conversation=conversation,
        )


def get_orchestrator(
    api_key: str = "demo-key",
) -> RubbishRAGOrchestrator:
    """Return the final orchestrator with the default System-1 judge."""

    return RubbishRAGOrchestrator(
        api_key=api_key
    )
