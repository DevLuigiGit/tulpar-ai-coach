
## router_rules — 2026-09-24 22:01

| arm | n | intent_accuracy | escalation_recall | false_escalation_rate | stability | p50_ms | cost_usd |
|---|---|---|---|---|---|---|---|
| rules | 66 | 78.8 | 72.2 | 6.2 | 100.0 | 0.0 | 0 |

## qa_local — 2026-09-24 22:01

| arm | n | embedder | rerank | temperature | top_p | pdf_chunk | hit_at_4 | mrr | keyfact_accuracy | correct_refusal_rate | faithfulness_avg | correctness_avg | p50_ms | out_tokens_p95 | cost_per_question_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local | 56 | local | True | 0.2 | 0.9 | 800 | 60.8 | 0.549 | 0.0 | 100.0 | None | None | 2.0 | 0 | 0.0 |

## exp_chunk_size_local — 2026-09-24 22:01

| arm | n | embedder | rerank | temperature | top_p | pdf_chunk | hit_at_4 | mrr | keyfact_accuracy | correct_refusal_rate | faithfulness_avg | correctness_avg | p50_ms | out_tokens_p95 | cost_per_question_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pdf_chunk=400, judge=False | 56 | local | True | 0.2 | 0.9 | 400 | 60.8 | 0.549 | 0.0 | 100.0 | None | None | 2.0 | 0 | 0.0 |
| pdf_chunk=800, judge=False | 56 | local | True | 0.2 | 0.9 | 800 | 60.8 | 0.549 | 0.0 | 100.0 | None | None | 2.0 | 0 | 0.0 |
| pdf_chunk=1200, judge=False | 56 | local | True | 0.2 | 0.9 | 1200 | 60.8 | 0.549 | 0.0 | 100.0 | None | None | 1.0 | 0 | 0.0 |

## exp_rag_rerank_local — 2026-09-24 22:01

| arm | n | embedder | rerank | temperature | top_p | pdf_chunk | hit_at_4 | mrr | keyfact_accuracy | correct_refusal_rate | faithfulness_avg | correctness_avg | p50_ms | out_tokens_p95 | cost_per_question_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rag_rerank=True | 56 | local | True | 0.2 | 0.9 | 800 | 60.8 | 0.549 | 0.0 | 100.0 | None | None | 2.0 | 0 | 0.0 |
| rag_rerank=False | 56 | local | False | 0.2 | 0.9 | 800 | 56.9 | 0.489 | 0.0 | 100.0 | None | None | 1.0 | 0 | 0.0 |

## qa_local — 2026-09-24 22:06

| arm | n | embedder | rerank | temperature | top_p | pdf_chunk | hit_at_4 | mrr | keyfact_accuracy | correct_refusal_rate | faithfulness_avg | correctness_avg | p50_ms | out_tokens_p95 | cost_per_question_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local | 56 | local | True | 0.2 | 0.9 | 800 | 60.8 | 0.549 | 0.0 | 100.0 | None | None | 2.0 | 0 | 0.0 |

## router_llm — 2026-09-24 23:31

| arm | n | intent_accuracy | escalation_recall | false_escalation_rate | stability | p50_ms | cost_usd |
|---|---|---|---|---|---|---|---|
| llm | 66 | 97.0 | 94.4 | 2.1 | 97.0 | 1345.5 | 0.077827 |

## draft — 2026-09-24 23:32

| arm | n | first_try_valid | final_valid | avg_drafts | p50_ms | cost_usd |
|---|---|---|---|---|---|---|
| default | 8 | 62.5 | 100.0 | 1.5 | 6126.0 | 0.0483 |

## qa_jina — 2026-09-24 23:33

| arm | n | embedder | rerank | temperature | top_p | pdf_chunk | hit_at_4 | mrr | keyfact_accuracy | correct_refusal_rate | faithfulness_avg | correctness_avg | p50_ms | out_tokens_p95 | cost_per_question_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jina | 56 | jina3 | True | 0.2 | 0.9 | 800 | 82.4 | 0.748 | 76.5 | 100.0 | 4.9 | 4.77 | 3304.0 | 232 | 0.000771 |

## draft_v1 — 2026-09-24 23:33

| arm | n | first_try_valid | final_valid | request_respected | avg_drafts | p50_ms | cost_usd |
|---|---|---|---|---|---|---|---|
| v1 | 9 | 77.8 | 88.9 | 88.9 | 1.33 | 3146 | 0.048936 |

## draft_v2 — 2026-09-24 23:35

| arm | n | first_try_valid | final_valid | request_respected | avg_drafts | p50_ms | cost_usd |
|---|---|---|---|---|---|---|---|
| v2 | 9 | 77.8 | 100.0 | 100.0 | 1.33 | 8274 | 0.049597 |

## vision_kimi-k2.7-code — 2026-09-24 23:38

| arm | n | models | top1 | top3 | errors | p50_ms |
|---|---|---|---|---|---|---|
| kimi-k2.7-code | 80 | ollama:kimi-k2.7-code | 72.5 | 78.8 | 7 | 4296.0 |

## router_llm_v2 — 2026-09-24 23:39

| arm | n | intent_accuracy | escalation_recall | false_escalation_rate | stability | p50_ms | cost_usd |
|---|---|---|---|---|---|---|---|
| llm_v2 | 66 | 100.0 | 100.0 | 0.0 | 100.0 | 1607.5 | 0.090218 |

## qa_jina_answer_v2 — 2026-09-24 23:42

| arm | n | embedder | rerank | temperature | top_p | pdf_chunk | hit_at_4 | mrr | keyfact_accuracy | correct_refusal_rate | faithfulness_avg | correctness_avg | p50_ms | out_tokens_p95 | cost_per_question_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jina_answer_v2 | 56 | jina3 | True | 0.2 | 0.9 | 800 | 82.4 | 0.768 | 86.3 | 100.0 | 4.57 | 4.51 | 4082.5 | 247 | 0.000828 |

## vision_minimax-m3 — 2026-09-24 23:43

| arm | n | models | top1 | top3 | errors | p50_ms |
|---|---|---|---|---|---|---|
| minimax-m3 | 80 | ollama:minimax-m3 | 63.8 | 71.2 | 5 | 3587.5 |

## exp_rag_rerank_jina — 2026-09-24 23:46

| arm | n | embedder | rerank | temperature | top_p | pdf_chunk | hit_at_4 | mrr | keyfact_accuracy | correct_refusal_rate | faithfulness_avg | correctness_avg | p50_ms | out_tokens_p95 | cost_per_question_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rag_rerank=True | 56 | jina3 | True | 0.2 | 0.9 | 800 | 80.4 | 0.748 | 62.7 | 100.0 | 4.82 | 4.56 | 4082.5 | 238 | 0.000759 |
| rag_rerank=False | 56 | jina3 | False | 0.2 | 0.9 | 800 | 84.3 | 0.758 | 76.5 | 100.0 | 4.83 | 4.67 | 2546.0 | 241 | 0.000765 |

## qa_jina_answer_v3 — 2026-09-24 23:50

| arm | n | embedder | rerank | temperature | top_p | pdf_chunk | hit_at_4 | mrr | keyfact_accuracy | correct_refusal_rate | faithfulness_avg | correctness_avg | p50_ms | out_tokens_p95 | cost_per_question_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jina_answer_v3 | 56 | jina3 | True | 0.2 | 0.9 | 800 | 80.4 | 0.725 | 78.4 | 100.0 | 4.78 | 4.54 | 3264.0 | 225 | 0.000873 |
