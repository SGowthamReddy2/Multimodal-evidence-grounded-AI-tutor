# 📊 Automated RAG Benchmark Evaluation Report

**Generated At:** 2026-11-22 01:48:44  
**Total Queries Evaluated:** 40  
**Skipped Queries (Rate Limit >= 60s):** 2  
**Overall Pass Rate (Config A):** 82.5%  

---

## 1. Executive Milestone Summary (M1 – M8)

| Evaluation Dimension | Milestone | Metric | Target | Result (Config A) | Status |
|---|---|---|---|---|:---:|
| **Citation Validity** | M3, NFR-01 | Fabricated Citation Rate | 0.0% | 0.0% | ✅ PASS |
| **Citation Grounding** | M3, NFR-02 | Citation Precision | >= 90% | 100.0% | ✅ PASS |
| **Abstention Accuracy**| M6, NFR-04 | Off-Corpus Abstention | >= 95% | 87.5% | ⚠️ WARN |
| **Semantic Alignment** | M1, M2, M4 | Cosine Similarity (BGE) | >= 0.70 | 0.8287 | ✅ PASS |
| **Response Latency (p50)** | M8, NFR-07 | Median Query Latency | <= 3500 ms | 24416 ms | ⚠️ WARN |
| **Response Latency (p95)** | M8, NFR-07 | Tail Query Latency | <= 5000 ms | 42851 ms | ⚠️ WARN |

---

## 2. Category Breakdown Performance

| Category | Queries | Passed | Pass Rate | Avg Latency | Avg Similarity |
|---|:---:|:---:|:---:|:---:|:---:|
| `text_only` | 6 | 5 | 83.3% | 20961.6 ms | 0.804 |
| `visual_required` | 6 | 5 | 83.3% | 32404.5 ms | 0.7353 |
| `multi_context` | 6 | 5 | 83.3% | 34308.5 ms | 0.8411 |
| `context_assembly` | 6 | 6 | 100.0% | 19127.8 ms | 0.8706 |
| `llm_generation` | 5 | 3 | 60.0% | 8961.8 ms | 0.7697 |
| `validation_gate` | 5 | 5 | 100.0% | 2656.0 ms | 1.0 |
| `frontend_presentation` | 6 | 4 | 66.7% | 24181.5 ms | 0.7988 |

---

## 3. Milestone 8: Config A vs. Config B Pareto Frontier Analysis

> **Config A (Quality-Optimized)**: Multi-channel RRF + Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)  
> **Config B (Speed-Optimized)**: Multi-channel RRF directly (bypassing Cross-Encoder reranker)

| Trade-off Dimension | Config A (Quality) | Config B (Speed) | Delta (A vs B) |
|---|---|---|---|
| **Pass Rate** | 82.5% | 82.86% | -0.4% |
| **Average Similarity** | 0.8287 | 0.8269 | +0.0018 |
| **Token F1** | 0.3138 | 0.3181 | -0.0043 |
| **Latency p50 (Median)** | 24415.7 ms | 28065.5 ms | -3649.8 ms |
| **Latency p95 (Tail)** | 42851.2 ms | 48979.9 ms | -6128.7 ms |

---

## 4. Query-by-Query Execution Details

| ID | Category | Decision | Latency | Sim | Citations Valid | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `TC_TXT_01` | `text_only` | `QUALIFIED` | 5555 ms | 0.846 | 100% | ✅ |
| `TC_TXT_02` | `text_only` | `FULL` | 8407 ms | 0.876 | 100% | ✅ |
| `TC_TXT_03` | `text_only` | `QUALIFIED` | 25503 ms | 0.751 | 100% | ❌ |
| `TC_TXT_04` | `text_only` | `FULL` | 29382 ms | 0.755 | 100% | ✅ |
| `TC_TXT_05` | `text_only` | `FULL` | 28858 ms | 0.861 | 100% | ✅ |
| `TC_TXT_06` | `text_only` | `FULL` | 28064 ms | 0.735 | 100% | ✅ |
| `TC_VIS_01` | `visual_required` | `FULL` | 42842 ms | 0.860 | 100% | ✅ |
| `TC_VIS_02` | `visual_required` | `FULL` | 35003 ms | 0.682 | 100% | ✅ |
| `TC_VIS_03` | `visual_required` | `FULL` | 31595 ms | 0.839 | 100% | ✅ |
| `TC_VIS_04` | `visual_required` | `FULL` | 29638 ms | 0.765 | 100% | ✅ |
| `TC_VIS_05` | `visual_required` | `ABSTAIN` | 23328 ms | 0.569 | 100% | ❌ |
| `TC_VIS_06` | `visual_required` | `FULL` | 32022 ms | 0.697 | 100% | ✅ |
| `TC_CTX_01` | `multi_context` | `FULL` | 37525 ms | 0.881 | 100% | ✅ |
| `TC_CTX_02` | `multi_context` | `FULL` | 26990 ms | 0.901 | 100% | ✅ |
| `TC_CTX_03` | `multi_context` | `ABSTAIN` | 2219 ms | 0.668 | 100% | ❌ |
| `TC_CTX_04` | `multi_context` | `FULL` | 40701 ms | 0.802 | 100% | ✅ |
| `TC_CTX_05` | `multi_context` | `QUALIFIED` | 55384 ms | 0.888 | 100% | ✅ |
| `TC_CTX_06` | `multi_context` | `FULL` | 43032 ms | 0.907 | 100% | ✅ |
| `TC_ASM_01` | `context_assembly` | `FULL` | 13345 ms | 0.848 | 100% | ✅ |
| `TC_ASM_02` | `context_assembly` | `FULL` | 33809 ms | 0.894 | 100% | ✅ |
| `TC_ASM_03` | `context_assembly` | `FULL` | 32330 ms | 0.903 | 100% | ✅ |
| `TC_ASM_04` | `context_assembly` | `FULL` | 27091 ms | 0.884 | 100% | ✅ |
| `TC_ASM_05` | `context_assembly` | `FULL` | 4518 ms | 0.894 | 100% | ✅ |
| `TC_ASM_06` | `context_assembly` | `FULL` | 3674 ms | 0.800 | 100% | ✅ |
| `TC_GEN_01` | `llm_generation` | `FULL` | 17766 ms | 0.868 | 100% | ✅ |
| `TC_GEN_03` | `llm_generation` | `FULL` | 3993 ms | 0.830 | 100% | ✅ |
| `TC_GEN_04` | `llm_generation` | `FULL` | 18429 ms | 0.891 | 100% | ✅ |
| `TC_GEN_05` | `llm_generation` | `ABSTAIN` | 1967 ms | 0.588 | 100% | ❌ |
| `TC_GEN_06` | `llm_generation` | `ABSTAIN` | 2655 ms | 0.671 | 100% | ❌ |
| `TC_VAL_01` | `validation_gate` | `ABSTAIN` | 2464 ms | 1.000 | 100% | ✅ |
| `TC_VAL_02` | `validation_gate` | `ABSTAIN` | 2293 ms | 1.000 | 100% | ✅ |
| `TC_VAL_03` | `validation_gate` | `ABSTAIN` | 2490 ms | 1.000 | 100% | ✅ |
| `TC_VAL_05` | `validation_gate` | `ABSTAIN` | 2476 ms | 1.000 | 100% | ✅ |
| `TC_VAL_06` | `validation_gate` | `ABSTAIN` | 3557 ms | 1.000 | 100% | ✅ |
| `TC_FNT_01` | `frontend_presentation` | `QUALIFIED` | 20210 ms | 0.823 | 100% | ✅ |
| `TC_FNT_02` | `frontend_presentation` | `FULL` | 32279 ms | 0.803 | 100% | ✅ |
| `TC_FNT_03` | `frontend_presentation` | `ABSTAIN` | 2216 ms | 0.651 | 100% | ❌ |
| `TC_FNT_04` | `frontend_presentation` | `FULL` | 29234 ms | 0.811 | 100% | ✅ |
| `TC_FNT_05` | `frontend_presentation` | `FULL` | 39306 ms | 0.835 | 100% | ❌ |
| `TC_FNT_06` | `frontend_presentation` | `FULL` | 21844 ms | 0.870 | 100% | ✅ |
