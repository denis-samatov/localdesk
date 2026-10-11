# Executed comparative benchmark

Evidence: `public_real_it_tickets_zenodo_7648117`. Test rows: 524.

Conclusion: **incomplete_qwen_comparison**. TF-IDF reference selected on tuning: `tfidf_svm`.

| Method | Status | Accuracy | Macro F1 | ECE | Target-policy coverage | Accepted error upper 95% | Inference p50 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | complete | 34.73% | 8.59% | 0.81% | 0.00% | unavailable | 0.002 |
| hashed_logistic | complete | 69.47% | 64.92% | 4.94% | 0.00% | unavailable | 0.029 |
| tfidf_logistic | complete | 77.67% | 73.98% | 2.76% | 0.00% | unavailable | 0.607 |
| tfidf_svm | complete | 77.10% | 73.54% | 16.69% | 0.00% | unavailable | 0.461 |
| qwen_zero_shot | not_run | — | — | — | — | — | — |
| qwen_decision | not_run | — | — | — | — | — | — |

## Paired group bootstrap (candidate minus reference)

- `hashed_logistic_vs_tfidf_svm`: macro-F1 delta -8.62 pp; 95% interval [-12.60, -4.81] pp.

## Limits

These reused public holdouts are not independent blind or field evaluations. Missing GPU stages do not demonstrate a Qwen advantage. Zero-coverage accuracy/error bounds are unavailable. The selected calibration Wilson bound is descriptive; heldout bounds assume independent rows. Latency is warmed local batch=1, not HTTP/load or monetary savings. See docs/BENCHMARK.md for the exact protocol and resource limitations.

Protocol SHA-256: `917b5d8649738683cb1b44b504d09bf2075b55dcd91762ae18e8ea5ce1f22d90`.
