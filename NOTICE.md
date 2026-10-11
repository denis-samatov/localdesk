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

Optional Unsloth / Unsloth Zoo dependencies and the Qwen model retain their own
licenses. No Unsloth implementation code or notebook content is copied into this
repository. API integration does not relicense the upstream software. Consult the
upstream licenses before redistribution or service deployment.
