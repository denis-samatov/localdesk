# Saved Qwen through the real HTTP API

On 2026-10-12 the IT production recipe completed two full epochs on a free Colab
Tesla T4 using core source `4d1082d51aa3966b7aaece301afb14dda40970af`.
The candidate `qwen-20261012-api` passed the unchanged IT gates. The saved adapter
and calibrated decision head were then loaded in separate offline and Uvicorn
processes. All 524 public heldout inputs were scored through actual HTTP requests.

Evidence: [GPU HTTP report](../reports/gpu-http-it.json),
[training report](../reports/qwen-serving-training.json),
[exact executed verifier](../scripts/fixtures/verify_gpu_http_20261012.py.txt).
The verifier's recorded SHA-256 matches the retained fixture byte for byte. The
small CLI wrapper validates that fingerprint before launching the fixture.

| Measurement | In-memory training evaluation | Saved/reloaded HTTP service |
| --- | ---: | ---: |
| Test count | 524 | 524 |
| Accuracy | 83.78% | 83.78% |
| Macro F1 | 81.44% | 81.44% |
| ECE | 1.78% | 1.71% |
| Routed count | 397 | 396 |
| Routing coverage | 75.76% | 75.57% |
| Routed accuracy | 91.94% | 91.92% |

All original gates passed on the HTTP outputs too. One decision moved to review
on saved-head reload; serialized precision and probability presentation can affect
threshold-boundary decisions. Do not assume identical probability/review metrics
from in-memory training and deployment. Three selected public cases additionally
matched exactly between fresh offline, single HTTP and batch HTTP inference.
Unauthenticated access returned 401; empty text, extra fields and oversized batches
returned 422. Backend identity was `unsloth`, with matching task and version hashes.
The verifier shut down only its own temporary loopback server.

This is a reused public corpus qualification, not a field deployment, load test,
Russian-language evaluation or a 5% operational error guarantee. It is a separate
serving run with 170 calibration rows, distinct from the 85-row calibration in the
comparative benchmark. Do not reuse that benchmark's confidence intervals for
this newly trained head.

## Weight recovery boundary

Colab created a complete 174,091,475-byte registry ZIP with SHA-256
`84d48aa10a04cc1c6e7d372452ad11a1a930f010fe4c90415a51522d44d5b6e9`.
During the local transfer, free GPU usage quota disconnected the runtime.
Only the first 138,412,032 bytes were preserved. Connecting without GPU allocated
a fresh empty CPU runtime. The untransferred part of the classification head was
lost; no complete local or durable-cloud checkpoint is claimed for this run.

The complete 129,934,448-byte LoRA adapter was recovered from the verified ZIP
prefix, with ZIP CRC checked and an independent SHA-256 recorded locally. Its
SHA-256 is `fb818993eb8f45a2386be2efb527349cbab9c84d918222ca4f80b960e0d79a5f`.
The adapter alone cannot reproduce these classifier scores or be served as a
LocalDesk decision model. Missing head weights must not be invented or replaced
with random initialization. Model binaries and the incomplete archive are excluded
from Git publication.

## Durable manual retry

Use [train_qwen_drive.ipynb](../notebooks/train_qwen_drive.ipynb).
The user approved Drive access, and the notebook's Drive permission prompt was
accepted. Actual Drive mounting/read-write verification did not complete in the
interrupted browser session. The new recipe has **not completed a GPU run**.
Its Python cells and deterministic regeneration were checked locally. No automatic
quota monitor or scheduled retraining was created; the user chose manual return.

1. Connect Drive and complete the write/fsync/rename/read probe before training.
2. Select free T4 after Google restores availability. CPU cannot train this recipe.
3. Train with the existing pipeline, using a new registry folder directly in Drive.
   Head, adapter and provenance persist there before test qualification completes.
4. Verify every artifact checksum, create the ZIP and SHA-256 sidecar in Drive,
   and confirm the archive exists in the Drive web interface before leaving Colab.
5. Only then run the same real HTTP verifier, retaining its report in Drive too.

Core source and the exact verifier Git blob are pinned independently. The notebook
fetches repository objects and refuses a verifier SHA mismatch. It creates new
session/version folders and does not overwrite or share existing Drive files.
The default local CPU registry remains unchanged. The GPU adapter also requires
the pinned base model cached on a Linux/CUDA serving machine.

For an already complete qualified IT artifact and prepared matching experiment:

```bash
python scripts/verify_it_qwen_http.py --config experiments/it-drive.json \
  --run-id YOUR_RUN_ID --out artifacts/http-verification-new
```

Use a new output directory for each verification. This fixture deliberately uses
the dedicated `qwen-api-registry` and three IT labels; the service core remains
configurable for other tasks. The archive/notebook are reproducible workflow
artifacts, not evidence that the interrupted model export was completed.
