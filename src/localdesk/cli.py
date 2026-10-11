"""Command-line entry points; the default learned baseline requires no GPU."""

import argparse
import json
import sys
from pathlib import Path

from localdesk.baseline import BaselineBackend
from localdesk.core import decide
from localdesk.data import load_splits
from localdesk.metrics import evaluate_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("predict", "serve", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--backend", choices=("baseline", "unsloth"), default="baseline")
        command.add_argument("--model", type=Path)
        if name == "predict":
            command.add_argument("text")
        elif name == "serve":
            command.add_argument("--port", type=int, default=8765)
        else:
            command.add_argument("--data", type=Path, default=Path("data/zenodo"))
    args = parser.parse_args()
    if args.backend == "unsloth" and not args.model:
        parser.error("--backend unsloth requires --model /path/to/local/checkpoint")
    if args.backend == "baseline" and args.model:
        parser.error("--model is only used with --backend unsloth")
    try:
        if args.backend == "baseline":
            backend = BaselineBackend()
        else:
            from localdesk.model import ModelBackend

            backend = ModelBackend(args.model)
        if args.command == "predict":
            print(json.dumps(decide(args.text, backend), ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            print(json.dumps(evaluate_rows(load_splits(args.data)["test"], backend), indent=2))
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
