# Reusable training plan

1. **Define the task.** Stable task ID/version, label descriptions, instructions
   and review policy. Use human-reviewed categories with enough distinct groups.
2. **Register data provenance.** Source/permission/license, raw checksums,
   text/label/id/conversation columns. User-provided provenance remains a claim
   until independently verified; adapters pin known public source bytes.
3. **Validate and split.** Check types, unknown/conflicting labels, duplicates,
   conversation links and equivalent inputs. Freeze train/calibration/test with
   observed class support. Preserve whole groups; fail on insufficient data.
4. **Fit a CPU model.** The recommended `tfidf` backend hashes word 1–2 grams and
   within-word character 3–5 grams into 2^18 buckets, keeps the most frequent
   buckets (`max_features` per block), applies sublinear TF-IDF and balanced
   logistic regression. The legacy `baseline` uses 4,096 hashed word buckets.
   Exported scoring must match sklearn on calibration data (float32 tolerance
   for `tfidf`). This establishes a cheap measured reference.
5. **Train Qwen if needed.** In a separate CUDA environment change backend to
   `unsloth`. LoRA rank, epochs, batches and learning rate are explicit; TaskSpec
   labels enter the decision head. Stop on skipped/truncated rows. Record GPU and
   dependency/source revisions. Smoke steps are not full-training qualification.
6. **Calibrate.** Fit temperature on the separate calibration split. No test
   labels participate in parameters, temperature or policy selection.
7. **Evaluate once.** Accuracy, macro/weighted F1, confusion, per-class metrics,
   NLL/Brier/ECE, coverage/accepted accuracy and a descriptive risk-coverage curve. Measure OOD/language robustness
   separately with independently labelled data, not confidence alone.
8. **Apply declared gates.** Failed/ungated versions are candidates. Keep metric
   values and failure reasons; do not hide failures or invent samples to pass.
9. **Publish the immutable artifact.** Task, data manifest, plan, report, payload,
   checksums and completion marker. Every model/report uses the same task and
   split fingerprints. Do not include private input texts or secrets.
10. **Promote and serve.** Explicitly choose a qualified version, restart workers,
    verify readiness and a task-specific request. Roll back by promoting an older
    qualified version and restarting. Confirm metrics against actual field labels
    before declaring operational accuracy.

The CLI emits a concrete `training-plan.json` for each completed run. Task,
backend parameters, seed, policy, gates and stage list are tied to that version.
The two published real-domain reports demonstrate configuration reuse; they do
not establish Qwen superiority, language generalization or field qualification.
