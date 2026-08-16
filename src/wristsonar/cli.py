"""Small, dependency-free command line for data integrity operations."""

from __future__ import annotations

import argparse
import socket
import sys
from collections.abc import Sequence
from pathlib import Path

from wristsonar.data.manifest import Manifest
from wristsonar.data.watchhand import build_watchhand_manifest


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be in 1..65535")
    return port


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


def _listen(args: argparse.Namespace) -> int:
    """Run the one-device raw PCM listener and emit pose JSON Lines to stdout."""
    from wristsonar.capture import LiveCaptureProcessor, consume_raw_pcm_connection
    from wristsonar.capture.wire import RawPcmWireFrame
    from wristsonar.model.load import load_torch_checkpoint
    from wristsonar.runtime import JsonLinesSink, RealtimeInference

    loaded = load_torch_checkpoint(args.weights, args.bundle)
    metadata = loaded.bundle.metadata
    processor = LiveCaptureProcessor()
    inference = RealtimeInference(
        predict=loaded.predictor,
        sink=JsonLinesSink(sys.stdout),
        model_id=loaded.model_id,
        calibration_minutes=metadata.calibration_minutes,
        expected_shape=(2, metadata.crop_bins, metadata.window_frames),
    )
    predicted = 0

    def process(packet: RawPcmWireFrame) -> None:
        nonlocal predicted
        for capture in processor.push(packet):
            inference.process(capture)
            predicted += 1

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        print(
            f"listening for one Wristsonar watch on {args.host}:{args.port}",
            file=sys.stderr,
        )
        connection, peer = listener.accept()
        with connection:
            packets = consume_raw_pcm_connection(connection, process)
        print(
            f"closed {peer[0]}:{peer[1]} after {packets} raw packets and "
            f"{predicted} pose frames",
            file=sys.stderr,
        )
    return 0


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

    live = commands.add_parser("live", help="run the verified host listener")
    live_actions = live.add_subparsers(dest="action", required=True)
    listen = live_actions.add_parser(
        "listen", help="accept one Wear OS raw PCM session and emit pose JSONL"
    )
    listen.add_argument("--weights", type=_path, required=True)
    listen.add_argument("--bundle", type=_path, required=True)
    listen.add_argument("--host", default="127.0.0.1")
    listen.add_argument("--port", type=_port, default=8766)
    listen.set_defaults(handler=_listen)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))
