"""Command-line entry points; the default learned baseline requires no GPU."""

import argparse
import json
import sys
from pathlib import Path

from localdesk.baseline import BaselineBackend
from localdesk.config import Experiment
from localdesk.core import decide
from localdesk.data import load_splits
from localdesk.dataset import prepare
from localdesk.metrics import evaluate_rows
from localdesk.registry import Registry
from localdesk.tfidf import demo_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("predict", "serve", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--backend", choices=("tfidf", "baseline", "unsloth"), default=None)
        command.add_argument("--model", type=Path)
        command.add_argument("--config", type=Path)
        if name == "predict":
            command.add_argument("text")
        elif name == "serve":
            command.add_argument("--port", type=int, default=8765)
        else:
            command.add_argument("--data", type=Path, default=Path("data/zenodo"))
    for name in ("validate", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "train":
            command.add_argument("--run-id")
    for name in ("runs", "promote"):
        command = subparsers.add_parser(name)
        command.add_argument("--registry", type=Path, default=Path("artifacts/registry"))
        if name == "promote":
            command.add_argument("run_id")
    command = subparsers.add_parser("benchmark")
    command.add_argument("config", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--qwen", action="store_true")
    command.add_argument("--checkpoint", type=Path)
    command.add_argument("--target-error", type=float, default=0.05)
    command.add_argument("--bootstrap-samples", type=int, default=2000)
    command = subparsers.add_parser("api")
    command.add_argument("--registry", type=Path, default=Path("artifacts/registry"))
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8765)
    command.add_argument("--anonymous-local", action="store_true")
    args = parser.parse_args()
    if args.command in {"validate", "train", "runs", "promote", "api", "benchmark"}:
        try:
            if args.command == "benchmark":
                from localdesk.benchmark import run_benchmark

                report = run_benchmark(
                    Experiment.load(args.config),
                    args.out,
                    qwen=args.qwen,
                    target_error=args.target_error,
                    bootstrap_samples=args.bootstrap_samples,
                    checkpoint=args.checkpoint,
                )
                print(
                    json.dumps(
                        {
                            "report": str(args.out / "report.json"),
                            "conclusion": report["conclusion"],
                        },
                        indent=2,
                    )
                )
                if args.qwen and report["conclusion"]["status"] == "incomplete_qwen_comparison":
                    raise SystemExit(2)
            elif args.command in {"validate", "train"}:
                experiment = Experiment.load(args.config)
                if args.command == "validate":
                    print(json.dumps(prepare(experiment).manifest, indent=2))
                else:
                    from localdesk.pipeline import run_experiment

                    print(json.dumps(run_experiment(experiment, run_id=args.run_id), indent=2))
            elif args.command == "runs":
                print(json.dumps(Registry(args.registry).list_runs(), indent=2))
            elif args.command == "promote":
                Registry(args.registry).promote(args.run_id)
                print(json.dumps({"active": args.run_id}))
            else:
                import uvicorn

                from localdesk.api import create_app

                if args.anonymous_local and args.host not in {"127.0.0.1", "::1", "localhost"}:
                    parser.error("anonymous serving requires a loopback host")
                uvicorn.run(
                    create_app(registry=args.registry, anonymous_local=args.anonymous_local),
                    host=args.host,
                    port=args.port,
                    access_log=False,
                )
            return
        except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
            print(f"LocalDesk: {error}", file=sys.stderr)
            raise SystemExit(1) from error
    if args.backend == "unsloth" and not args.model:
        parser.error("--backend unsloth requires --model /path/to/local/checkpoint")
    try:
        if args.model and (args.model / "manifest.json").is_file():
            backend, _ = Registry(args.model.resolve().parent).load(
                args.model.name, require_qualified=args.command != "evaluate"
            )
            if args.backend and backend.name != args.backend:
                parser.error("backend flag differs from artifact backend")
        elif args.backend in {None, "tfidf", "baseline"}:
            if args.model:
                parser.error("model path must be a completed registry artifact")
            # Packaged demo models; the legacy v0.1 baseline stays selectable.
            backend = BaselineBackend() if args.backend == "baseline" else demo_backend()
        else:
            from localdesk.model import ModelBackend

            backend = ModelBackend(args.model)
        if args.command == "predict":
            print(json.dumps(decide(args.text, backend), ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            if args.config:
                experiment = Experiment.load(args.config)
                if not backend.task or backend.task.digest != experiment.task.digest:
                    raise ValueError("evaluation task differs from model task")
                rows = prepare(experiment).rows["test"]
            else:
                rows = load_splits(args.data)["test"]
            print(json.dumps(evaluate_rows(rows, backend), indent=2))
        else:
            from localdesk.server import make_server

            server = make_server(backend, port=args.port)
            print(f"LocalDesk ({backend.name}): http://127.0.0.1:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            finally:
                server.server_close()
    except KeyboardInterrupt:
        return
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"LocalDesk: {error}", file=sys.stderr)
        raise SystemExit(1) from error
