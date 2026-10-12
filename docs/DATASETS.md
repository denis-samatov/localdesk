# Real datasets and reusable intake

## IT Support Tickets

Leonardo Santiago Benitez Pereira, *Classification of IT Support Tickets*,
Zenodo v2, 2022. https://doi.org/10.5281/zenodo.7648117 — CC BY 4.0.
Original: 2,229 company requests with seven human-assigned source labels. The
author describes multilingual text and PII removal through automated tools,
regex and manual review. We do not claim independent PII or annotation audits.

The adapter verifies four CSV hashes and joins labels by ID. Grouping considers
actual baseline features, removes conflicting groups and keeps test representatives
of duplicates crossing the official split. EOL has only two distinct automated
texts and is excluded from supported choices; both are kept as a separate challenge,
not an adequate OOD detector evaluation. Prepared supported splits: 690/170/524.
The generic service importer rechecks group isolation; it never inserts IT labels.

## SMS Spam Collection

Tiago Almeida and Jos Hidalgo, UCI, 2011. https://doi.org/10.24432/C5CC84 — CC BY 4.0.
Public messages assembled from research/public sources; legitimate messages include
voluntary public contributions. Labels are original `ham` and `spam`. The downloaded
text has 5,574 rows; the UCI description rounds/reports its own source count.

`prepare_sms.py` verifies the fixed ZIP and contained text SHA-256 before CSV
conversion. Generic intake collapses 448 equivalent representations, leaving
5,126 examples split into 3334 train / 768 calibration / 1024 test with seed3407.
No invented messages, translations or labels enter fitting/evaluation.
The corpus is historical; it does not represent all contemporary spam or languages.

## Banking77

Iñigo Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*,
2020. https://github.com/PolyAI-LDN/task-specific-datasets — CC BY 4.0.
Online banking queries with 77 fine-grained intents; 10,003 train and 3,080 test rows.

`prepare_banking77.py` downloads the CSVs at a pinned commit and verifies their
SHA-256. Calibration is a deterministic, label-stratified 15% of the official
training rows (seed 3407); the official test set is kept. Model-equivalent
duplicates are collapsed so no group crosses splits (a test copy wins) and
conflicting labels are dropped: 8,434 train / 1,493 calibration / 3,071 test.
Label descriptions are derived from the source intent names; no texts or labels
are invented.

## CLINC150

Stefan Larson et al., *An Evaluation Dataset for Intent Classification and
Out-of-Scope Prediction*, EMNLP 2019. https://github.com/clinc/oos-eval — CC BY 3.0.
Virtual-assistant queries for 150 intents across ten domains, plus queries that
match none of them.

`prepare_clinc150.py` uses the pinned `data_full.json`: official train, validation
(as calibration) and test splits for the 150 in-scope intents, after the same
duplicate isolation: 14,877 train / 2,983 calibration / 4,486 test. The 1,000
official out-of-scope test queries are written to `ood.jsonl`; they never enter
training, calibration or in-scope metrics and are only used to measure how often
the model sends them to review.

## New data

Choose CSV or JSONL and map text/label/id/conversation columns in Experiment JSON.
`id_column` is optional but supplied IDs must be unique; otherwise stable positional
IDs are assigned. `group_column` is optional. Dotted paths apply to nested JSON
objects; literal column keys (including dots in CSV headers) are checked first.
Source data are limited to 100 MB per file for this single-node importer.
Text is single-label, 1–4000 characters. Labels must match the TaskSpec exactly.

One-file mode computes grouped train/calibration/test splits. Three-file mode
requires exact train/calibration/test keys and isolated groups; it does not silently
repair caller-owned split leakage. Conflicting equivalent-text labels are errors.
Rare classes need sufficient distinct groups; unsupported target types need an adapter.

Source hashes, split hashes, duplicate counts, class support and seed are stored.
Raw texts and private paths are not included in model/report manifests. Dataset
provenance strings provided by a user are not independent evidence of origin,
license or correctness. Protect private source files and registry access under
your organization's policies. Source adapters here download only public material.
