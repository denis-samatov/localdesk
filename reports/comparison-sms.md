# Executed comparative benchmark

Evidence: `public_real_sms_uci_228`. Test rows: 1024.

Conclusion: **incomplete_qwen_comparison**. TF-IDF reference selected on tuning: `tfidf_logistic`.

| Method | Status | Accuracy | Macro F1 | ECE | Target-policy coverage | Accepted error upper 95% | Inference p50 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | complete | 87.79% | 46.75% | 0.98% | 0.00% | unavailable | 0.001 |
| hashed_logistic | complete | 97.46% | 93.82% | 0.62% | 100.00% | 3.69% | 0.022 |
| tfidf_logistic | complete | 98.93% | 97.41% | 0.54% | 100.00% | 1.91% | 0.427 |
| tfidf_svm | complete | 98.93% | 97.41% | 12.85% | 99.80% | 1.79% | 0.431 |
| qwen_zero_shot | not_run | — | — | — | — | — | — |
| qwen_decision | not_run | — | — | — | — | — | — |

## Paired group bootstrap (candidate minus reference)

- `hashed_logistic_vs_tfidf_logistic`: macro-F1 delta -3.59 pp; 95% interval [-5.58, -1.89] pp.

## Limits

These reused public holdouts are not independent blind or field evaluations. Missing GPU stages do not demonstrate a Qwen advantage. Zero-coverage accuracy/error bounds are unavailable. The selected calibration Wilson bound is descriptive; heldout bounds assume independent rows. Latency is warmed local batch=1, not HTTP/load or monetary savings. See docs/BENCHMARK.md for the exact protocol and resource limitations.

Protocol SHA-256: `7cf3aef7c85a9c18d19c362e409aefccd852d711045076a64f37f970d9bec492`.
