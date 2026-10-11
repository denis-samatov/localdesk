"""Download only the four public CSVs; never execute upstream pickle/code artifacts."""

import argparse
import json
import urllib.request
from pathlib import Path

from localdesk.data import (
    SOURCE,
    SOURCE_HASHES,
    build_splits,
    load_splits,
    read_source,
    write_dataset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/zenodo"))
    parser.add_argument("--out", type=Path, default=Path("data/zenodo"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.source.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_HASHES:
        target = args.source / name
        if not target.exists():
            if args.offline:
                parser.error(f"missing cached file: {name}")
            with urllib.request.urlopen(
                f"{SOURCE}/files/{name}?download=1", timeout=60
            ) as response:
                payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ValueError("unexpected source file size")
            target.write_bytes(payload)
    rows, audit = build_splits(read_source(args.source))
    write_dataset(args.out, rows, audit, SOURCE_HASHES)
    load_splits(args.out)
    print(
        json.dumps(
            {
                "counts": {split: len(values) for split, values in rows.items()},
                "cross_split_groups": audit["cross_split_groups"],
                "conflicting_groups": audit["conflicting_groups"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
