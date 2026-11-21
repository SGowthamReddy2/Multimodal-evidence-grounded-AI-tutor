# 📊 Automated RAG Benchmark Evaluation Report

**Generated At:** 2026-11-21 23:52:05  
**Total Queries Evaluated:** 42  
**Overall Pass Rate (Config A):** 73.81%  

---

## 1. Executive Milestone Summary (M1 – M8)

| Evaluation Dimension | Milestone | Metric | Target | Result (Config A) | Status |
|---|---|---|---|---|:---:|
| **Citation Validity** | M3, NFR-01 | Fabricated Citation Rate | 0.0% | 0.0% | ✅ PASS |
| **Citation Grounding** | M3, NFR-02 | Citation Precision | >= 90% | 100.0% | ✅ PASS |
| **Abstention Accuracy**| M6, NFR-04 | Off-Corpus Abstention | >= 95% | 78.6% | ⚠️ WARN |
| **Semantic Alignment** | M1, M2, M4 | Cosine Similarity (BGE) | >= 0.70 | 0.6390 | ⚠️ WARN |
| **Response Latency (p50)** | M8, NFR-07 | Median Query Latency | <= 3500 ms | 2395 ms | ✅ PASS |
| **Response Latency (p95)** | M8, NFR-07 | Tail Query Latency | <= 5000 ms | 4405 ms | ✅ PASS |

---

## 2. Category Breakdown Performance

| Category | Queries | Passed | Pass Rate | Avg Latency | Avg Similarity |
|---|:---:|:---:|:---:|:---:|:---:|
| `text_only` | 6 | 5 | 83.3% | 3443.5 ms | 0.6864 |
| `visual_required` | 6 | 3 | 50.0% | 2337.1 ms | 0.4936 |
| `multi_context` | 6 | 5 | 83.3% | 3143.2 ms | 0.6257 |
| `context_assembly` | 6 | 6 | 100.0% | 2894.3 ms | 0.5652 |
| `llm_generation` | 6 | 4 | 66.7% | 2185.3 ms | 0.5022 |
| `validation_gate` | 6 | 6 | 100.0% | 2452.6 ms | 1.0 |
| `frontend_presentation` | 6 | 2 | 33.3% | 2546.6 ms | 0.6001 |

---

## 3. Milestone 8: Config A vs. Config B Pareto Frontier Analysis

> **Config A (Quality-Optimized)**: Multi-channel RRF + Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)  
> **Config B (Speed-Optimized)**: Multi-channel RRF directly (bypassing Cross-Encoder reranker)

| Trade-off Dimension | Config A (Quality) | Config B (Speed) | Delta (A vs B) |
|---|---|---|---|
| **Pass Rate** | 73.81% | 85.71% | -11.9% |
| **Average Similarity** | 0.6390 | 0.4840 | +0.1550 |
| **Token F1** | 0.2401 | 0.1204 | +0.1197 |
| **Latency p50 (Median)** | 2395.3 ms | 367.4 ms | +2027.9 ms |
| **Latency p95 (Tail)** | 4404.8 ms | 1659.1 ms | +2745.7 ms |

---

## 4. Query-by-Query Execution Details

| ID | Category | Decision | Latency | Sim | Citations Valid | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `TC_TXT_01` | `text_only` | `FULL` | 4787 ms | 0.906 | 100% | ✅ |
| `TC_TXT_02` | `text_only` | `FULL` | 3873 ms | 0.868 | 100% | ✅ |
| `TC_TXT_03` | `text_only` | `QUALIFIED` | 4432 ms | 0.766 | 100% | ❌ |
| `TC_TXT_04` | `text_only` | `FULL` | 2967 ms | 0.819 | 100% | ✅ |
| `TC_TXT_05` | `text_only` | `FULL` | 2146 ms | 0.394 | 100% | ✅ |
| `TC_TXT_06` | `text_only` | `FULL` | 2455 ms | 0.364 | 100% | ✅ |
| `TC_VIS_01` | `visual_required` | `ABSTAIN` | 2572 ms | 0.607 | 100% | ❌ |
| `TC_VIS_02` | `visual_required` | `FULL` | 2295 ms | 0.439 | 100% | ✅ |
| `TC_VIS_03` | `visual_required` | `FULL` | 2283 ms | 0.404 | 100% | ❌ |
| `TC_VIS_04` | `visual_required` | `FULL` | 2359 ms | 0.476 | 100% | ✅ |
| `TC_VIS_05` | `visual_required` | `ABSTAIN` | 2040 ms | 0.590 | 100% | ❌ |
| `TC_VIS_06` | `visual_required` | `FULL` | 2473 ms | 0.447 | 100% | ✅ |
| `TC_CTX_01` | `multi_context` | `FULL` | 5143 ms | 0.825 | 100% | ✅ |
| `TC_CTX_02` | `multi_context` | `FULL` | 3833 ms | 0.917 | 100% | ✅ |
| `TC_CTX_03` | `multi_context` | `ABSTAIN` | 2304 ms | 0.668 | 100% | ❌ |
| `TC_CTX_04` | `multi_context` | `FULL` | 2735 ms | 0.383 | 100% | ✅ |
| `TC_CTX_05` | `multi_context` | `QUALIFIED` | 2450 ms | 0.468 | 100% | ✅ |
| `TC_CTX_06` | `multi_context` | `FULL` | 2394 ms | 0.493 | 100% | ✅ |
| `TC_ASM_01` | `context_assembly` | `FULL` | 2464 ms | 0.426 | 100% | ✅ |
| `TC_ASM_02` | `context_assembly` | `FULL` | 2397 ms | 0.418 | 100% | ✅ |
| `TC_ASM_03` | `context_assembly` | `FULL` | 3879 ms | 0.884 | 100% | ✅ |
| `TC_ASM_04` | `context_assembly` | `FULL` | 3738 ms | 0.879 | 100% | ✅ |
| `TC_ASM_05` | `context_assembly` | `FULL` | 2663 ms | 0.456 | 100% | ✅ |
| `TC_ASM_06` | `context_assembly` | `FULL` | 2224 ms | 0.328 | 100% | ✅ |
| `TC_GEN_01` | `llm_generation` | `FULL` | 2575 ms | 0.510 | 100% | ✅ |
| `TC_GEN_02` | `llm_generation` | `QUALIFIED` | 2318 ms | 0.446 | 100% | ✅ |
| `TC_GEN_03` | `llm_generation` | `FULL` | 2123 ms | 0.357 | 100% | ✅ |
| `TC_GEN_04` | `llm_generation` | `FULL` | 2239 ms | 0.421 | 100% | ✅ |
| `TC_GEN_05` | `llm_generation` | `ABSTAIN` | 1987 ms | 0.588 | 100% | ❌ |
| `TC_GEN_06` | `llm_generation` | `ABSTAIN` | 1870 ms | 0.692 | 100% | ❌ |
| `TC_VAL_01` | `validation_gate` | `ABSTAIN` | 2236 ms | 1.000 | 100% | ✅ |
| `TC_VAL_02` | `validation_gate` | `ABSTAIN` | 1959 ms | 1.000 | 100% | ✅ |
| `TC_VAL_03` | `validation_gate` | `ABSTAIN` | 1948 ms | 1.000 | 100% | ✅ |
| `TC_VAL_04` | `validation_gate` | `QUALIFIED` | 3114 ms | 1.000 | 100% | ✅ |
| `TC_VAL_05` | `validation_gate` | `ABSTAIN` | 2065 ms | 1.000 | 100% | ✅ |
| `TC_VAL_06` | `validation_gate` | `ABSTAIN` | 3393 ms | 1.000 | 100% | ✅ |
| `TC_FNT_01` | `frontend_presentation` | `ABSTAIN` | 2102 ms | 0.650 | 100% | ❌ |
| `TC_FNT_02` | `frontend_presentation` | `FULL` | 2275 ms | 0.372 | 100% | ✅ |
| `TC_FNT_03` | `frontend_presentation` | `ABSTAIN` | 2183 ms | 0.650 | 100% | ❌ |
| `TC_FNT_04` | `frontend_presentation` | `QUALIFIED` | 3275 ms | 0.785 | 100% | ✅ |
| `TC_FNT_05` | `frontend_presentation` | `ABSTAIN` | 2131 ms | 0.674 | 100% | ❌ |
| `TC_FNT_06` | `frontend_presentation` | `ABSTAIN` | 3313 ms | 0.469 | 100% | ❌ |
