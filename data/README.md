# Data card

**Source:** Leonardo Santiago Benitez Pereira. *Classification of IT Support Tickets*.
Zenodo, 2022, v2. https://doi.org/10.5281/zenodo.7648117 — CC BY 4.0.
The authors describe real company support tickets, three human annotators, and
PII removal through AWS Comprehend, regular expressions and manual review.
We rely on that source provenance; we did not independently re-audit every text
for residual personal information.

Only these source files are downloaded: X_train.csv, y_train.csv, X_test.csv,
y_test.csv. Their SHA-256 hashes are embedded in `localdesk.data.SOURCE_HASHES`.
We never download or execute the source's pickle, model or Python artifacts.

Preparation retains original text and labels, joins by ID, and detects normalized
(NFKC, case-folded, whitespace-collapsed) duplicate text groups and groups with equivalent model features.
Feature-equivalent grouping also accounts for PII-tag removal, tokenization,
hashing and L2 normalization, rounded to 12 decimal places. Groups with
contradictory labels are excluded. Remaining groups keep one row, preferring the
official test representative. A 20% per-class calibration split, rounded down,
comes exclusively from the remaining official training pool, with seed 3407.
No translation or augmentation is performed.

Counts: 690 train, 170 calibration, 524 test, 2 EOL challenge texts. The published
source has 2,229 rows; the adapted corpus has 1,386 unique consistent texts.
Only six source categories are supported. EOL's two distinct templates are
excluded from the model's choices and never used for training or calibration.

The source does not contain urgency, action, OOD-detector or language labels.
All application review decisions are derived policy, not new human gold labels.
Russian quality cannot be established from this corpus. Related-but-feature-distinct tickets may remain: this conservative grouping
is not semantic or temporal deduplication. The official split is not a time-held-out or organization-held-out
production evaluation.

Run `python scripts/prepare_data.py` from the repository root. It writes ignored
raw data to `data/raw/zenodo/` and converted records plus a manifest to
`data/zenodo/`. Every prepared record contains state, questions, gold, original
ID, source label/split and a model-input group hash. `load_splits` verifies
schema, labels, file hashes and group isolation before training/evaluation.
Use `--offline` to require cached source files.

For a local organization's validation, collect authorized, anonymized requests,
keep related conversations together, use independent reviewed category labels,
reserve separate calibration/test data and measure class recall, review coverage,
accepted accuracy and OOD false accepts before changing routing policy.

```bibtex
@dataset{benitez_pereira_2022_it_tickets,
  author = {Leonardo Santiago Benitez Pereira},
  title = {Classification of IT Support Tickets},
  year = {2022},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.7648117},
  url = {https://doi.org/10.5281/zenodo.7648117}
}
```
