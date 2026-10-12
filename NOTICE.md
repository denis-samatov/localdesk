The project code and original notebook are MIT licensed.

The training data are **Classification of IT Support Tickets**, Leonardo
Santiago Benitez Pereira, Federal Institute of Santa Catarina, 2022, v2.
Source: https://doi.org/10.5281/zenodo.7648117
License: https://creativecommons.org/licenses/by/4.0/

Changes made by this project: ID-based CSV joining; normalization for duplicate
detection (NFKC, case folding, whitespace collapse); conservative grouping by
PII-tag-stripped, tokenized and hashed baseline input features; duplicate and conflicting
label removal; a calibration split from source training data; source EOL templates
reserved as an unsupported-category challenge; conversion to Unsloth choice rows.
The published linear baseline was trained on this adapted data. Keep this attribution
with the baseline. Ticket texts are downloaded on demand and excluded from this Git repository.

The second configured domain uses **SMS Spam Collection**, Tiago Almeida and
Jos Hidalgo, UCI Machine Learning Repository, 2011.
Source: https://doi.org/10.24432/C5CC84 — CC BY 4.0.
Changes: verified archive extraction and conversion to id/text/label CSV;
conservative equivalent-input deduplication and deterministic grouped splits.
Original labels are retained. Raw SMS content is excluded from Git and serving logs.

The third and fourth domains are downloaded on demand and excluded from Git:

**Banking77**, Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson
and Ivan Vulić, "Efficient Intent Detection with Dual Sentence Encoders", 2020.
Source: https://github.com/PolyAI-LDN/task-specific-datasets — CC BY 4.0.

**CLINC150**, Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke,
Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano,
Lingjia Tang and Jason Mars, "An Evaluation Dataset for Intent Classification and
Out-of-Scope Prediction", EMNLP 2019. Source: https://github.com/clinc/oos-eval —
CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/).

Changes to both: verified extraction to id/text/label JSONL; collapsing of
model-equivalent duplicates so no group crosses splits; dropping of conflicting
labels; a deterministic calibration split from Banking77 training data; CLINC150
out-of-scope test queries kept in a separate file. Original labels are retained.

Optional Unsloth / Unsloth Zoo dependencies and the Qwen model retain their own
licenses. No Unsloth implementation code or notebook content is copied into this
repository. API integration does not relicense the upstream software. Consult the
upstream licenses before redistribution or service deployment.
