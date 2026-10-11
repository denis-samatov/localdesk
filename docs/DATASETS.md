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
