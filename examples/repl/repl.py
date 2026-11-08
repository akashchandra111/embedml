#!/usr/bin/env python3
"""Drive ``elml_run --repl`` from Python without reloading the model.

The model is loaded once by the C process. Each feature row is written as one
line to its stdin, and one JSON line is read from its stdout. The same protocol
can later be carried over a serial link by an MCU firmware.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def default_runner() -> Path:
    return Path(__file__).resolve().parents[2] / "build" / "elml_run"


def format_reply(reply: dict[str, Any]) -> str:
    if reply.get("task") == "classification":
        lines = [
            f"{entry['label']}  {entry['probability']:.6f}"
            for entry in reply.get("top", [])
        ]
        return "\n".join(lines)
    lines = [
        f"{entry['name']} = {entry['value']:g}"
        for entry in reply.get("outputs", [])
    ]
    return "\n".join(lines)


class Runner:
    def __init__(self, executable: Path, model: Path, top: int | None) -> None:
        command = [str(executable), str(model), "--repl", "--json"]
        if top is not None:
            command.extend(["--top", str(top)])
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def predict(self, row: str) -> dict[str, Any]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(row + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = ""
            if self.process.stderr is not None:
                error = self.process.stderr.read().strip()
            raise RuntimeError(
                f"elml_run exited with {self.process.wait()}: {error}"
            )
        return json.loads(line)

    def close(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
            self.process.stdin.close()
        self.process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="trained .elml model")
    parser.add_argument(
        "rows",
        nargs="*",
        help="comma-separated feature rows; omit for an interactive loop",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(os.environ.get("ELML_RUN", default_runner())),
        help="path to elml_run (default: build/elml_run or $ELML_RUN)",
    )
    parser.add_argument("--top", type=int, default=None, help="show top-K classes")
    args = parser.parse_args(argv)

    runner = Runner(args.runner, args.model, args.top)
    try:
        if args.rows:
            for row in args.rows:
                print(format_reply(runner.predict(row)))
        else:
            print("Enter comma-separated features, or 'quit'.", file=sys.stderr)
            while True:
                try:
                    row = input("features> ")
                except EOFError:
                    break
                if row.strip().lower() in {"quit", "exit", ":quit", ":exit"}:
                    break
                if not row.strip():
                    continue
                try:
                    print(format_reply(runner.predict(row)))
                except (RuntimeError, json.JSONDecodeError) as error:
                    print(f"error: {error}", file=sys.stderr)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
