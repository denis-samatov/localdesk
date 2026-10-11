"""Export reviewed aggregate reports only; never export model data or per-row predictions."""

import argparse
import json
from pathlib import Path

from localdesk.config import digest, read_json
from localdesk.registry import atomic_json


def percent(value):
    return "unavailable" if value is None else f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    report = read_json(args.run / "report.json")
    if report["protocol_sha256"] != digest(report["protocol"]):
        raise ValueError("report protocol fingerprint mismatch")
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.prefix.with_suffix(".json"), report)
    lines = [
        "# Executed comparative benchmark",
        "",
        f"Evidence: `{report['protocol']['evidence']}`. "
        f"Test rows: {report['protocol']['split_counts']['test']}.",
        "",
        f"Conclusion: **{report['conclusion']['status']}**. "
        f"TF-IDF reference selected on tuning: `{report['reference']}`.",
        "",
        "| Method | Status | Accuracy | Macro F1 | ECE | Target-policy coverage | "
        "Accepted error upper 95% | Inference p50 (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in (
        "majority",
        "hashed_logistic",
        "tfidf_logistic",
        "tfidf_svm",
        "qwen_zero_shot",
        "qwen_decision",
    ):
        method = report["methods"][name]
        if method["status"] != "complete":
            lines.append(f"| {name} | {method['status']} | — | — | — | — | — | — |")
            continue
        metrics, policy = method["classification"], method["target_policy_test"]
        lines.append(
            f"| {name} | complete | {percent(metrics['accuracy'])} | "
            f"{percent(metrics['macro_f1'])} | {percent(metrics['ece'])} | "
            f"{percent(policy['coverage'])} | {percent(policy['error_upper_95'])} | "
            f"{method['latency_ms']['p50']:.3f} |"
        )
    lines.extend(["", "## Paired group bootstrap (candidate minus reference)", ""])
    for name, comparison in report["comparisons"].items():
        interval = comparison["macro_f1_delta_ci95"]
        lines.append(
            f"- `{name}`: macro-F1 delta {comparison['macro_f1_delta'] * 100:+.2f} pp; "
            f"95% interval [{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}] pp."
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "These reused public holdouts are not independent blind or field evaluations. "
            "Missing GPU stages do not demonstrate a Qwen advantage. "
            "Zero-coverage accuracy/error bounds are unavailable. The selected calibration "
            "Wilson bound is descriptive; heldout bounds assume independent rows. "
            "Latency is warmed local batch=1, not HTTP/load or monetary savings. "
            "See docs/BENCHMARK.md for the exact protocol and resource limitations.",
            "",
            f"Protocol SHA-256: `{report['protocol_sha256']}`.",
            "",
        ]
    )
    args.prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(args.prefix.with_suffix(".json")),
                "markdown": str(args.prefix.with_suffix(".md")),
            }
        )
    )


if __name__ == "__main__":
    main()
