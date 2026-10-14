# Engineering Evidence Report (PROOF.md)

> This report documents the final RubbishRAG investigation, architectural changes, negative results, and measured behavior. Results are reported only on the evaluation path that actually exercises the corresponding feature. Visible benchmark, paraphrase DEV, controlled robustness sets, and Judge splits are not treated as interchangeable.

---

## Candidate Information

- **Candidate Name:** Fateme Fouladi
- **Submission Date:** 2026-10-14
- **Execution Target:** Windows 10 / Python virtual environment / CPU
- **Track Scope Completed:**
  - [x] Required Core
  - [x] Elective E: Evidence Diversity
  - [x] Elective F: Adversarial / Noise-Document Defense
  - [x] Elective G: Negation / Polarity-Aware Matching
  - [x] Elective H: Grounded Clarification
  - [x] Staff Track: Adaptive System-1 Decision Judge

---

## Executive Summary & Scorecard

The starter system had several independent failure modes rather than one bad parameter: incomplete candidate recall, unstable sparse/dense fusion, Persian surface-form drift, adversarial evidence, polarity mismatch, evidence redundancy, and under/over-clarification.

The final candidate-controlled architecture uses:

1. deterministic Persian normalization and structured slot extraction;
2. boundary-aware, metadata-preserving chunking;
3. conservative gated RRF in the remote branch;
4. a complementary **Question-only local BM25 retriever**;
5. Remote + Local candidate union without adding incompatible raw scores;
6. trusted-corpus evidence rehydration and grounding-aware reranking;
7. polarity-aware matching and document-level evidence diversity;
8. intent-sensitive clarification;
9. an Adaptive System-1 Judge for routing, context budgeting, sufficiency, stopping, and abstention;
10. a thin orchestrator that reuses the canonical pipeline instead of duplicating logic.

### Scorecard

| Metric | Baseline | Final Result | Delta | Target | Evaluation Source |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Overall Accuracy** | 30.0% | **40.0%** | +10 pp | ≥65% | Last successful official visible `bench` |
| **Retrieval Recall@5** | 20.0% | **75.0%** | +55 pp | ≥75% | Non-exact paraphrase DEV: remote-only → final hybrid |
| **Candidate-Pool Recall@20** | 45.0% | **90.0%** | +45 pp | — | Non-exact paraphrase DEV |
| **Clarification Accuracy** | 0.0% | **100.0% (8/8)** | +100 pp | ≥50% | Controlled ambiguity set |
| **Mean Query Latency** | Not separately captured | **34.4 ms** | — | <100 ms | Last successful official visible `bench` |
| **Poison Top-3 Citation Slots** | 5.0% (3/60) | **0.0% (0/60)** | -5 pp | 0.0% | Controlled poison ablation |
| **Judge Routing Accuracy** | 93.3% | **100.0%** | +6.7 pp | — | Provided Judge test split |
| **Judge Clarification F1** | 0.400 | **1.000** | +0.600 | — | Provided Judge test split |
| **Automated Non-CLI Tests** | — | **51/51 passed** | — | All green | `pytest --ignore=tests/test_cli.py` |

### Evaluation-Path Caveat

The stock `python rubbish.py bench` path directly calls the sealed remote retriever and then the registered reranker/resolver. It does **not** exercise the registered candidate `@R.retriever`, so it does not measure the final Remote + Local BM25 candidate-generation path.

The last successful stock benchmark before remote quota exhaustion was:

```text
total        = 20
accuracy     = 0.40
answer_acc   = 0.40
clarify_acc  = 0.00
mean latency = 34.4 ms
p50          = 30.0 ms
p95          = 37.7 ms
```

The visible benchmark contains answer cases only, so `clarify_acc = 0.0` is not evidence that the clarification gate fails. Clarification was evaluated separately.

I did not modify `rubbish.py` or the sealed retriever to route the official benchmark through my custom retriever.

---

## Investigation 1: Baseline Failure Localization, Persian Normalization & Entity Sensitivity

### 1. Observed Symptom

The naive visible benchmark achieved:

| Metric | Baseline |
| :--- | :---: |
| End-to-End Accuracy | 30.0% (6/20) |
| Hit@3 | 40.0% (8/20) |
| Hit@5 | 55.0% (11/20) |
| Mean Top1–Top5 fused spread | 0.2974 |
| Queries with adversarial doc in raw Top-3 | 20.0% (4/20) |

Failure decomposition:

- 6/20 produced a correct final answer;
- 2/20 had gold evidence in raw Top-3 but still failed downstream;
- 3/20 retrieved gold only at ranks 4–5;
- 9/20 missed gold entirely in Top-5.

This showed that retrieval recall was the largest bottleneck, but not the only one.

### 2. Falsifiable Hypothesis

Equivalent Persian queries can tokenize differently because of Persian/ASCII digits, Arabic/Persian character variants, half-spaces, and colloquial forms. If this is a surface-form problem, deterministic canonicalization should improve controlled retrieval/reranking behavior without requiring a larger model.

### 3. Diagnostic Probes & Exact Trace References

The raw trace schema contains `ts`, `key`, `q`, `topk`, and `hits`, but no separate `trace_id` field. The provided `inspect-trace` CLI accepts the **JSONL line number** as an ID, so I use exact line numbers below. Each can be reproduced with `python rubbish.py inspect-trace <line>`.

| Probe | CLI Trace ID | Timestamp (`ts`) |
| :--- | :---: | :--- |
| `۴۸۰ شدم. چرا؟` | `88` | `1791636260.6244135` |
| explicit financial-conviction query | `89` | `1791636295.8679497` |
| `رتبه C3 یعنی چی؟` | `90` | `1791636306.6978285` |
| expanded 480 causal query | `92` | `1791637187.0443833` |
| explicit `480 → rank` query | `93` | `1791637206.5926666` |
| C2 rank probe | `95` | `1791638361.7642012` |
| ASCII `250` variant | `98` | `1791639269.9021127` |
| Persian `۲۵۰` variant | `99` | `1791639276.2133043` |
| `رتبه C1 چیست؟` | `100` | `1791639312.0422416` |
| `رتبه C1 چیه؟` | `101` | `1791639319.3412287` |

### 4. Quantitative Evidence

Normalization and slot evaluation:

- normalization correctness: **13/13**;
- idempotence: **10/10**;
- slot cases: **12/12**;
- controlled reranker Top-1: **25% → 100%**;
- regressions in that controlled reranker set: **0**.

The ASCII `250` and Persian `۲۵۰` probes produced materially different upstream candidate behavior, while `رتبه C1 چیست؟` and `رتبه C1 چیه؟` behaved identically. This supported **selective**, not universal, normalization sensitivity.

### 5. Root Cause Analysis

The starter relied heavily on raw surface overlap. It did not provide one canonical search representation for:

- Persian/Arabic digit scripts;
- Arabic `ي/ى/ك/ة` vs Persian forms;
- ZWNJ vs ordinary spaces;
- colloquial possessive/verb forms;
- bank aliases and A1–E3 rank formatting.

### 6. Architectural Modification

- **Modules:** `rubbish_rag/normalize_fa.py`, `rubbish_rag/slots.py`
- Persian / Arabic-Indic digits → ASCII;
- Arabic character variants → canonical Persian;
- ZWNJ → normal space for retrieval;
- longest-first, boundary-aware colloquial normalization;
- structured BANK, RANK, PERSONA, DEBT, AMOUNT/NUMS, TIME, NEGATED extraction;
- longest alias first for ambiguous bank names.

### 7. Ablation

The controlled reranker improved from **25% Top-1 to 100% Top-1** with the normalization/structured matching path. The normalizer also remained idempotent on all tested cases.

### 8. Production Trade-offs

The implementation is deterministic, dependency-free, and cheap. The key limitation is execution order: candidate-side normalization cannot remove script sensitivity inside the sealed upstream service when the stock harness sends the raw query there first.

---

## Investigation 2: Retrieval Fusion, Local BM25 & Hybrid Candidate Coverage

### 1. Observed Symptom

The visible remote retrieval baseline was:

- Hit@3: **40%**
- Hit@5: **55%**

Several candidates showed disagreement between sparse and dense rankings, but replacing the server ranking globally caused regressions.

### 2. Falsifiable Hypothesis

Rank fusion should help only when the raw score distribution is visibly unstable. Separately, an independent local Question index may recover paraphrases missed by the sealed remote retriever.

### 3. Diagnostic Probes & Exact Trace References

Representative retained traces:

- short numeric/intent failure: trace `88`;
- verbose causal reformulation: trace `92`;
- explicit 480-to-rank probe: trace `93`;
- financial-conviction case with relevant doc `56` below distractors: trace `89`.

The gated-RRF gain itself is reported from the saved ablation, not inferred from the minimal JSONL record, because `trace.jsonl` stores only final `fused` values rather than the separate BM25/dense sub-scores needed to recompute the gate.

### 4. Quantitative Evidence

#### Rejected global alternatives

- Global RRF: Hit@3 **40% → 40%**, Hit@5 **55% → 45%**
- Percentile normalization:
  - P75: 40% / 50%
  - P90: 30% / 55%
  - P95: 35% / 55%
- Length/BM25 mean Spearman: **0.0148**
- 5-fold BM25-heavy tuning: H@3 **35%**, H@5 **55%**, MRR **0.323**
- Untuned baseline: H@3 **40%**, H@5 **55%**, MRR **0.367**

#### Selected remote-branch gate

Apply RRF only when:

1. BM25 or dense contains a Tukey upper outlier; and
2. BM25 Top-1 and dense Top-1 disagree.

Result:

- gate activated: **7/20**;
- Hit@3: **40% → 45%**;
- Hit@5: **55% → 55%**;
- MRR: **0.367 → 0.371**;
- observed gold regressions: **0**.

#### Local retriever on non-exact paraphrase DEV

The visible benchmark queries were exact normalized matches to corpus Questions, so I did **not** use exact visible lookup as generalization evidence. Instead I created a 20-query non-exact paraphrase DEV set.

| Local Backend | H@1 | H@3 | H@5 | H@20 | MRR |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Count-cosine | 15% | 55% | 65% | 80% | 0.367 |
| TF-IDF | 45% | 70% | **85%** | 90% | 0.598 |
| BM25 | **55%** | **75%** | 80% | 90% | **0.658** |

#### Full hybrid backend ablation

| Local Backend | Pool Recall | H@1 | H@3 | H@5 | H@20 | MRR | Gold Cited Top-3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Count | 85% | 20% | 45% | 70% | 85% | 0.362 | 45% |
| TF-IDF | 90% | 50% | 60% | **75%** | 90% | 0.595 | 60% |
| **BM25** | **90%** | **60%** | **70%** | **75%** | **90%** | **0.667** | **70%** |

Final hybrid DEV:

- full-pool gold coverage: **18/20 = 90%**;
- after-reranking H@1: **60%**;
- H@3: **70%**;
- H@5: **75%**;
- H@20: **90%**;
- MRR: **0.667**;
- gold cited Top-3: **14/20 = 70%**.

Gold source distribution:

- local only: **9**;
- remote + local: **9**;
- missing from both: **2**.

### 5. Root Cause Analysis

A single sealed candidate source left recall gaps. Global score fusion also discarded useful score magnitude in some cases. The local Question-only representation provided complementary evidence without assuming its BM25 scale was directly comparable with the remote score.

### 6. Architectural Modification

- **Module:** `rubbish_rag/pipeline.py`
- conservative gated RRF in the remote branch;
- Question-only local BM25 (`k1=1.2`, `b=0.75`);
- punctuation-aware tokenization;
- Remote + Local candidate union by document ID;
- no raw cross-index score addition;
- best-source rank prior;
- final explicit reranking.

### 7. Ablation

Before vs after final reranking on the same hybrid DEV pool:

| Metric | Before | After |
| :--- | :---: | :---: |
| H@1 | 55% | **60%** |
| H@3 | 70% | 70% |
| H@5 | 75% | 75% |
| H@20 | 90% | 90% |
| MRR | 0.641 | **0.667** |

### 8. Production Trade-offs

The local BM25 branch is CPU-friendly and dependency-free. The 75% H@5 result is explicitly a **DEV result**, not an untouched hidden-test estimate. Two DEV golds remain outside the combined Top-20 pool.

---

## Investigation 3: Adversarial Grounding, Trusted Rehydration & Polarity

### 1. Observed Symptom

Injected or fabricated documents could rank highly through keyword stuffing or near-duplicate wording. A second problem appeared after hybridization: legitimate remote document IDs sometimes arrived with snippets that failed strict text-grounding against the canonical corpus.

### 2. Falsifiable Hypothesis

Evidence trust should be based on content/provenance rather than document-ID ranges. Known documents should be rehydrated from canonical trusted corpus text, while unknown evidence should remain untrusted.

### 3. Diagnostic Probes & Exact Trace References

- doc `9002` retrieved in the tax probe: trace `2`;
- doc `9106` retrieved in the financial-conviction probe: trace `3`;
- docs `9000` and `9003` retrieved together: trace `14`;
- doc `9004` ranked first for `رتبه C3 یعنی چی؟`: trace `17`;
- volitional negation (`نمی‌خوام ...`): trace `11`;
- factual negation (`هیچ سابقه تاخیری ندارد`): trace `20`;
- broader Top-20 financial-conviction stress probe with `9000/9106`: trace `729`;
- C3 paraphrase where `9004` remains in the candidate pool: trace `743`.

### 4. Quantitative Evidence

Grounding detector over 100 retrieved candidates:

- known poison: **6**
- genuine: **94**
- TP / FP / TN / FN: **6 / 0 / 94 / 0**
- precision: **1.00**
- recall: **1.00**
- specificity: **1.00**

Controlled poison ablation:

| Metric | Before | After |
| :--- | ---: | ---: |
| Queries with poison in Top-3 | 3/20 | **0/20** |
| Poison Top-3 citation slots | 3/60 | **0/60** |
| Gold Top-1 | 25% | **35%** |
| Gold Hit@3 | 40% | **40%** |

Hybrid grounding failure before rehydration:

```text
remote+local: 7 golds, verified 0/7, all 7 ranked worse
remote only : 2 golds, verified 0/2, both ranked worse
local only  : 7 golds, verified 7/7
```

After trusted rehydration:

```text
remote+local: 7/7 verified; 3 improved, 4 same, 0 worse
remote only : 2/2 verified; 1 improved, 1 same, 0 worse
local only  : 7/7 verified; 2 improved, 5 same, 0 worse
```

Polarity evaluation:

- **8/8 correct**
- false positives: **0**
- false negatives: **0**

### 5. Root Cause Analysis

`doc_id >= 9000` is not a valid production security rule. ID magnitude is not semantic evidence. Strict snippet-to-answer text comparison also incorrectly distrusted legitimate remote snippets.

### 6. Architectural Modification

- **Module:** `rubbish_rag/pipeline.py`
- trusted normalized Answer/BriefAnswer allow-list;
- known document IDs rehydrated with canonical trusted evidence;
- original snippet retained separately for diagnostics;
- unknown IDs remain `retrieved_only`;
- concept-level polarity checks around domain anchors;
- volitional language such as `نمی خواهم` excluded from factual-negation penalties;
- grounded and polarity-compatible evidence ranked first.

### 7. Ablation

Poison disappeared from every evaluated Top-3 citation slot while Hit@3 stayed unchanged. Rehydration then removed false distrust of legitimate remote golds without weakening the poison defense.

### 8. Production Trade-offs

The defense is local and auditable, but string/content grounding is not a complete adversarial-security model. A production system with external corpora should use stronger provenance/version metadata.

---

## Investigation 4: Boundary-Aware Chunking & Evidence Diversity

### 1. Observed Symptom

The baseline chunker sliced answers every 300 characters regardless of sentence structure. Multiple chunks from one document could also occupy evidence slots without adding new document-level information.

### 2. Falsifiable Hypothesis

Sentence-first chunking plus protected numeric phrases should reduce semantic boundary damage. Document-level diversity should improve unique evidence coverage without changing candidate count.

### 3. Diagnostic Probes & Exact Trace IDs

Chunking was primarily evaluated corpus-wide rather than by a single query trace.

Relevant supporting traces:

- keyword-heavy C3 candidate example: trace `17`;
- Top-20 financial-conviction stress probe: trace `729`;
- Top-20 bounced-check stress probe: trace `730`.

The chunk-quality ablation itself is corpus-level and is therefore supported primarily by `plots/chunking_ablation.png`, not by a single retrieval trace.

Primary artifact:
- `plots/chunking_ablation.png`

### 4. Quantitative Evidence

Across 442 documents:

| Metric | Fixed 300 | Boundary + Metadata |
| :--- | ---: | ---: |
| Chunks | 494 | 527 |
| Mean length | 200.0 | 256.7 |
| Median length | 214.5 | 264.0 |
| P95 length | 300 | 348 |
| Max length | 300 | 360 |
| Bad internal boundary rate | **98.2%** | **1.5%** |
| Intact sentence preservation | 82.2% | **94.7%** |
| Numeric phrase/unit preservation | 99.4% | **100.0%** |
| Retrieval-visible metadata coverage | 0.0% | **100.0%** |

Diversity ablation:

- duplicate-document queries in Top-5: **2/20**;
- duplicate slot rate: **2%**;
- average unique docs in Top-3: **2.95 → 3.00**;
- Hit@3: **40% → 40%**;
- gold-rank regressions: **0**.

### 5. Root Cause Analysis

Fixed-length slicing optimizes only size. It does not preserve sentence/clause structure or numeric spans such as amounts, score ranges, percentages, and time units.

### 6. Architectural Modification

- **Module:** `rubbish_rag/pipeline.py`
- sentence-first greedy packing;
- clause/word fallback;
- protected numeric expressions;
- `[Category | Keyword]` retrieval-visible prefix;
- category/keyword/question metadata retained structurally;
- max final chunk length 360;
- document diversity promotion after reranking.

### 7. Ablation

Bad internal boundaries dropped from **98.2% to 1.5%**, intact-sentence preservation increased from **82.2% to 94.7%**, and metadata coverage increased from **0% to 100%**.

### 8. Production Trade-offs

Chunk count increased by ~6.7%, and the maximum chunk length increased from 300 to 360 characters. I accepted that bounded cost because it materially improved semantic integrity.

---

## Investigation 5: Intent-Sensitive Clarification & Adaptive System-1 State Consistency

### 1. Observed Symptom

The baseline resolver answered from the top hit even when the user omitted information required to answer safely. A broad missing-slot rule then caused the opposite problem: false clarification on otherwise answerable queries.

The initial Judge also allowed pre- and post-retrieval states to contradict each other.

### 2. Falsifiable Hypothesis

Clarification should depend on **intent + decision-critical missing context**, not every absent slot. Retrieved evidence also cannot infer user information that was never supplied, so `CLARIFY` must remain `CLARIFY` after retrieval.

### 3. Diagnostic Probes & Exact Trace References

- generic debt query used to verify underspecification: trace `986`;
- bank-specified loan eligibility control (should remain answerable): trace `988`;
- specific causal repayment query that should remain answerable: trace `13`.

The short status-only clarification cases are handled **before retrieval**, so a remote retrieval trace is intentionally absent for them. They are reproduced through `judge-bench` / the local Judge diagnostic rather than by forcing an unnecessary `/retrieve` call.

Controlled local Judge cases included:

```text
می‌تونی کمکم کنی؟
رتبم اومده پایین چرا؟
وام گرفتم رتبه‌م تغییر نکرده
ضامن شدم
مالیات ندادم
درآمدم بالاست ولی امتیازم پایینه، چرا؟
```

### 4. Quantitative Evidence

Controlled ambiguity set:

| Metric | Baseline | Final |
| :--- | :---: | :---: |
| Overall status accuracy | 10/18 | **18/18** |
| Ambiguous queries clarified | 0/8 | **8/8** |
| Specific queries answered | 10/10 | **10/10** |

Visible answerability gate:

- false clarifications: **0/20**

Adaptive Judge:

| Metric | Initial Test | Final Test | Final Dev |
| :--- | :---: | :---: | :---: |
| Routing Accuracy | 93.3% | **100.0%** | **100.0%** |
| Clarification Precision | 50.0% | **100.0%** | **100.0%** |
| Clarification Recall | 33.3% | **100.0%** | **100.0%** |
| Clarification F1 | 0.400 | **1.000** | **1.000** |
| Mean Dynamic Top-k | 3.47 | **3.13** | 3.33 |
| Context Savings vs fixed k=5 | 30.7% | **37.3%** | 33.3% |
| Retrieval Brier | 0.0624 | **0.0022** | **0.0022** |
| Retrieval ECE | **0.022** | 0.046 | 0.046 |

Sufficiency state-machine diagnostic:

- before final consistency fix:
  - dev: **9/15 = 60.0%**
  - test: **10/15 = 66.7%**
- after fix:
  - dev: **15/15 = 100%**
  - test: **15/15 = 100%**

### 5. Root Cause Analysis

Two overly independent rules caused most failures:

1. `missing slot -> clarify` was too broad;
2. post-retrieval sufficiency could override a valid pre-retrieval clarification decision.

### 6. Architectural Modification

- **Modules:** `rubbish_rag/pipeline.py`, `rubbish_rag/judge.py`, `rubbish_rag/orchestrator.py`
- intent-sensitive clarification;
- conversational/meta routing to no-retrieval;
- dynamic budgets `{0, 2, 3, 5, 8}`;
- state consistency across pre/post retrieval;
- collective evidence sufficiency;
- early stopping;
- abstention on unsafe/conflicting evidence;
- thin orchestrator reusing canonical modules.

Final state machine:

```text
NO_RETRIEVAL                  -> SUFFICIENT -> ANSWER without retrieval
CLARIFY                       -> INSUFFICIENT -> CLARIFY
RETRIEVE + adequate evidence  -> SUFFICIENT -> ANSWER
unsafe/conflicting evidence   -> IRRELEVANT/CONFLICTING -> ABSTAIN
```

### 7. Ablation

State consistency alone moved sufficiency diagnostics from **60.0% / 66.7%** to **100% / 100%** on dev/test.

The final Judge also reduced mean context from fixed `k=5` to **3.13** on the provided test split, a **37.3% context reduction**.

### 8. Production Trade-offs

The Judge is deterministic and CPU-only. The main limitation is coverage: heuristic rules need a larger independently labeled routing/clarification set before production deployment.

> The provided Judge `test` split was inspected during debugging, so these numbers are benchmark/debugging evidence, not an untouched hold-out estimate.

---

## Deliberate Non-Interventions

1. **Rejected: Global RRF everywhere**
   - Hit@5 dropped from 55% to 45%; several queries regressed.

2. **Rejected: Percentile score normalization**
   - P75/P90/P95 variants did not beat the server baseline.

3. **Rejected: Generic length bonus**
   - Length/BM25 correlation was approximately zero.

4. **Rejected: Static BM25-heavy raw-score tuning**
   - The apparent full-set gain disappeared under deterministic cross-validation.

5. **Rejected: Equal multi-view local RRF**
   - Word, character, and metadata fusion reduced H@5 despite useful individual signals.

6. **Rejected: Visible exact-question lookup as generalization**
   - Visible queries were exact normalized matches to corpus Questions; using that result would be leakage.

7. **Rejected: Poison-ID hardcoding**
   - `doc_id >= 9000` is not a valid production trust rule.

8. **Rejected: Heavy Cross-Encoder / open-weight retrieval model**
   - Added CPU/dependency/model-selection cost was not justified by evidence under the submission constraints.

9. **Rejected: Broad `missing slot -> clarify`**
   - It produced false clarifications.

10. **Rejected: Modifying `rubbish.py` or sealed `server/*.pyc`**
    - That would change the evaluation contract rather than improve the candidate implementation.

---

## Bonus Track: Adaptive System-1 Decision Judge

- **Judge Architecture:** deterministic heuristic System-1 Judge with structured Persian slots, calibrated routing probabilities, intent-sensitive clarification, collective evidence checks, dynamic context budgets, adaptive stopping, and abstention.
- **Routing Accuracy:**
  - provided test: **100.0%**
  - dev: **100.0%**
- **Clarification F1:**
  - provided test: **1.000**
  - dev: **1.000**
- **Mean Dynamic Top-k:**
  - provided test: **3.13**
  - dev: **3.33**
- **Context Savings vs fixed k=5:**
  - provided test: **37.3%**
  - dev: **33.3%**
- **Retrieval Calibration:**
  - Brier Score: **0.0022**
  - ECE: **0.046**
- **Clarification Calibration:**
  - test Brier: **0.0040**
  - dev Brier: **0.0051**
  - test ECE: **0.0520**
  - dev ECE: **0.0587**
- **Sufficiency State Diagnostic:**
  - test: **15/15**
  - dev: **15/15**

**Key insight:** the Judge does not stop on one raw retrieval score. It preserves the pre-retrieval state, then checks trust, entity coverage, polarity compatibility, suspicious/repetitive content, and collective support. It stops when evidence is sufficient with confidence ≥0.80, otherwise it can expand once within the already retrieved pool or abstain/clarify.

---

## Production Architecture Summary

```text
                          +----------------------+
Query ------------------> | System-1 Judge       |
                          +----------+-----------+
                                     |
                  +------------------+------------------+
                  |                  |                  |
           NO_RETRIEVAL           CLARIFY            RETRIEVE
                  |                  |                  |
           direct response      targeted question      v
                                               +----------------+
                                               | Remote branch  |
                                               | + gated RRF    |
                                               +----------------+
                                                        \
                                                         \
                                               +----------------+
                                               | Local BM25     |
                                               | Question-only  |
                                               +----------------+
                                                         /
                                                        /
                                               +----------------+
                                               | Candidate union|
                                               +--------+-------+
                                                        |
                                               +--------v-------+
                                               | Trusted corpus |
                                               | rehydration    |
                                               +--------+-------+
                                                        |
                                               +--------v-------+
                                               | Reranker       |
                                               | grounding      |
                                               | polarity       |
                                               | diversity      |
                                               +--------+-------+
                                                        |
                                               +--------v-------+
                                               | Evidence Judge |
                                               | adaptive stop  |
                                               +---+--------+---+
                                                   |        |
                                                 answer   abstain/
                                                          clarify
```

---

## Current Limitations

1. **Official visible overall accuracy remains 40%.** The ≥65% official overall target is not demonstrated by the stock benchmark.
2. **Final hybrid Recall@5 = 75% is a DEV result**, not hidden or untouched hold-out performance.
3. Two paraphrase DEV queries remain outside the combined Remote + Local Top-20 pool.
4. The stock benchmark bypasses the registered custom retriever.
5. Final hybrid-path end-to-end latency was not separately rerun after remote quota exhaustion; 34.4 ms is the last successful stock-bench mean.
6. The provided Judge test split was inspected during debugging.
7. Full CLI smoke tests could not be rerun after `RuntimeError: quota exceeded`.

---

## Trace Evidence Index

The following distinct traces are used as reproducible evidence. Run, for example, `python rubbish.py inspect-trace 88`.

| Trace ID | Probe / Evidence |
| :---: | :--- |
| `2` | tax query retrieving poison doc 9002 |
| `3` | financial-conviction query retrieving 9106 |
| `11` | volitional negation probe |
| `13` | specific causal repayment query |
| `14` | raw retrieval containing 9000 and 9003 |
| `17` | C3 rank query with 9004 at Top-1 |
| `20` | factual negation / no-delay query with 9001 |
| `88` | short `۴۸۰ شدم. چرا؟` failure |
| `89` | financial-conviction rank structure |
| `90` | C3 poison-heavy retrieval |
| `92` | expanded 480 causal formulation |
| `93` | explicit 480-to-rank formulation |
| `95` | C2 entity probe |
| `98` | ASCII `250` |
| `99` | Persian `۲۵۰` |
| `100` | formal C1 wording |
| `101` | colloquial C1 wording |
| `729` | Top-20 financial-conviction adversarial stress probe |
| `730` | Top-20 bounced-check stress probe |
| `743` | C3 paraphrase with poison candidate deeper in pool |
| `986` | generic debt ambiguity probe |
| `988` | bank-specified loan eligibility control |

## Reproduction Checklist

Completed:

- [x] `python -m pytest tests/ -q --ignore=tests/test_cli.py`
  - **51 passed**
- [x] `python rubbish.py judge-bench --split test`
  - routing 100%, clarification F1 1.000, context savings 37.3%, Brier 0.0022
- [x] `python rubbish.py judge-bench --split dev`
  - routing 100%, clarification F1 1.000, context savings 33.3%, Brier 0.0022
- [x] `python -m analysis.diagnose_judge_sufficiency`
  - test 15/15, dev 15/15
- [x] Orchestrator import / no-retrieval / clarification smoke paths
- [x] Non-CLI regression suite after orchestrator cleanup
  - **51 passed**

Still required before packaging:

- [x] Exact CLI trace references inserted using JSONL line IDs accepted by `inspect-trace`.
- [x] At least 15 distinct retrieval trace IDs are cited across the report.
- [ ] Run `python -m analysis.make_final_plots`.
- [ ] Verify:
  - `plots/length_vs_bm25.png`
  - `plots/hybrid_ablation.png`
  - `plots/hybrid_rerank_ablation.png`
  - `plots/chunking_ablation.png`
  - `plots/judge_ablation.png`
- [ ] Run `python -m analysis.validate_submission_artifacts`.
- [ ] If remote quota is available, rerun full `pytest tests/` and `python rubbish.py bench`.
- [ ] Run `python rubbish.py bastesh`.
- [ ] Inspect `submissions/submission.zip`.

---

## Final Engineering Summary

The largest improvement did not come from a larger model or aggressive global retuning. It came from isolating failure modes and applying narrow changes supported by ablations.

The final retrieval architecture combines the sealed remote branch with a complementary Question-only BM25 view. On the non-exact paraphrase DEV set, full-pool coverage reached **90%**, final Hit@5 reached **75%**, and gold appeared in the final Top-3 citations for **14/20** queries.

Separately, trusted-corpus grounding removed observed poison evidence from evaluated Top-3 citations, rehydration fixed false distrust of legitimate remote snippets, polarity checks passed **8/8** controlled cases, boundary-aware chunking reduced bad internal boundaries from **98.2% to 1.5%**, and intent-sensitive clarification reached **18/18** controlled status accuracy.

The Adaptive System-1 Judge adds a coherent routing/sufficiency state machine with **100% routing accuracy**, **1.000 clarification F1**, and **37.3% context savings** on the provided test split, while the thin orchestrator removes the older duplicated production logic.

The remaining limitations are explicit: official stock overall accuracy is still **40%**, hybrid retrieval numbers are DEV rather than held-out results, the Judge test split was inspected during debugging, and the final remote CLI path could not be rerun after quota exhaustion.
