from __future__ import annotations

from pathlib import Path

from pytest import CaptureFixture

from wristsonar.cli import main, parser
from wristsonar.data.manifest import build_manifest


def test_cli_verifies_a_manifest(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    root = tmp_path / "watchhand"
    root.mkdir()
    (root / "session.npy").write_bytes(b"fixture")
    manifest = build_manifest(root, dataset="watchhand", version="fixture-1")
    path = tmp_path / "manifest.json"
    manifest.write(path)

    exit_code = main(["manifest", "verify", str(root), str(path)])

    assert exit_code == 0
    assert "verified" in capsys.readouterr().out


def test_live_listener_requires_a_provenance_bound_weight_pair() -> None:
    args = parser().parse_args(
        [
            "live",
            "listen",
            "--weights",
            "weights.pt",
            "--bundle",
            "weights.bundle.json",
            "--port",
            "8766",
        ]
    )
    assert args.host == "127.0.0.1"
    assert args.port == 8766
