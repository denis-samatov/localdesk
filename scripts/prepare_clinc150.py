"""Fetch the pinned CLINC150 "full" file, verify it, and write isolated JSONL splits.

The 150 in-scope intents use the official train / validation (as calibration) / test
splits. The 1,000 official out-of-scope test queries are written separately to
ood.jsonl; they never enter training, calibration or the in-scope test metrics.
"""

import argparse
import json
from pathlib import Path

from localdesk.sources import fetch_verified, isolate_splits, write_jsonl_splits

COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{COMMIT}/data/data_full.json"
SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
SPLITS = {"train": "train", "val": "calibration", "test": "test", "oos_test": "ood"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/raw/clinc150/data_full.json"))
    parser.add_argument("--out", type=Path, default=Path("data/clinc150"))
    args = parser.parse_args()
    source = json.loads(fetch_verified(URL, SHA256, limit=4_000_000, cache=args.cache))
    records = []
    for name, split in SPLITS.items():
        for index, (text, label) in enumerate(source[name]):
            if (label == "oos") != (split == "ood"):
                raise ValueError(f"{name}: unexpected out-of-scope label placement")
            records.append(
                {"id": f"{name}-{index:05d}", "text": text, "label": label, "split": split}
            )
    # Out-of-scope queries are isolated too, so none duplicates an in-scope example.
    splits, audit = isolate_splits(records, preference=("ood", "test", "calibration", "train"))
    manifest = write_jsonl_splits(
        args.out,
        splits,
        {
            "source": "https://github.com/clinc/oos-eval",
            "commit": COMMIT,
            "source_sha256": {"data_full.json": SHA256},
            "license": "CC-BY-3.0",
            "audit": audit,
        },
    )
    print(json.dumps({"counts": audit["counts"], "sha256": manifest["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
