
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
