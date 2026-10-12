"""Fetch the licensed UCI corpus, verify it, and convert its original labels to CSV."""

import argparse
import csv
import hashlib
import io
import zipfile
from pathlib import Path

from localdesk.sources import read_bounded

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_SHA = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
TEXT_SHA = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="use this cached ZIP without network")
    parser.add_argument("--out", type=Path, default=Path("data/sms/rows.csv"))
    args = parser.parse_args()
    payload = args.source.read_bytes() if args.source else read_bounded(URL, 1_000_000)
    if hashlib.sha256(payload).hexdigest() != ZIP_SHA:
        raise ValueError("UCI archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        data = archive.read("SMSSpamCollection")
    if hashlib.sha256(data).hexdigest() != TEXT_SHA:
        raise ValueError("UCI text checksum mismatch")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "text", "label"])
        writer.writeheader()
        for index, line in enumerate(data.decode("utf-8").splitlines()):
            label, text = line.split("\t", 1)
            if label not in {"ham", "spam"}:
                raise ValueError("unexpected UCI label")
            writer.writerow({"id": f"sms-{index}", "text": text, "label": label})
    print(f"Converted {index + 1} original rows; source license CC BY 4.0")


if __name__ == "__main__":
    main()
