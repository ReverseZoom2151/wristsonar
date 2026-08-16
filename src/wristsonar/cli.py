"""Small, dependency-free command line for data integrity operations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from wristsonar.data.manifest import Manifest
from wristsonar.data.watchhand import build_watchhand_manifest


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _build(args: argparse.Namespace) -> int:
    manifest = build_watchhand_manifest(
        args.root, version=args.version, notes=args.notes
    )
    manifest.write(args.output)
    print(f"wrote {args.output}: {len(manifest)} files, {manifest.total_bytes} bytes")
    return 0


def _verify(args: argparse.Namespace) -> int:
    manifest = Manifest.read(args.manifest)
    report = manifest.verify(args.root, allow_extra=args.allow_extra)
    print(report.describe())
    return 0 if report.ok else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="wristsonar")
    commands = root.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser(
        "manifest", help="build or verify WatchHand identity"
    )
    actions = manifest.add_subparsers(dest="action", required=True)

    build = actions.add_parser("build", help="hash a downloaded WatchHand tree")
    build.add_argument("root", type=_path)
    build.add_argument("--version", required=True)
    build.add_argument("--output", required=True, type=_path)
    build.add_argument("--notes", default="")
    build.set_defaults(handler=_build)

    verify = actions.add_parser("verify", help="verify a tree against its manifest")
    verify.add_argument("root", type=_path)
    verify.add_argument("manifest", type=_path)
    verify.add_argument("--allow-extra", action="store_true")
    verify.set_defaults(handler=_verify)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))
