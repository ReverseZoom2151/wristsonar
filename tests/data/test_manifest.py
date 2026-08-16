"""The integrity layer's job is to say no. These tests make it say no."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wristsonar.data.manifest import (
    MANIFEST_SCHEMA,
    FileRecord,
    IntegrityError,
    LazyVerifier,
    Manifest,
    build_manifest,
    sha256_file,
)

_SUFFIXES = (".npy", ".json", ".txt", ".csv")


def _manifest(root: Path) -> Manifest:
    return build_manifest(
        root, dataset="watchhand", version="test-1", include_suffixes=_SUFFIXES
    )


def test_build_records_only_wanted_suffixes(watchhand_root: Path) -> None:
    manifest = _manifest(watchhand_root)
    paths = [record.path for record in manifest.files]

    assert paths, "fixture produced no manifestable files"
    assert all(Path(p).suffix in _SUFFIXES for p in paths)
    assert not any(p.endswith(".DS_Store") or p.endswith(".png") for p in paths)
    assert paths == sorted(paths), "manifest order must be stable for diffing"
    assert all("\\" not in p for p in paths), "paths must be posix on every platform"


def test_clean_tree_verifies(watchhand_root: Path) -> None:
    report = _manifest(watchhand_root).verify(watchhand_root, allow_extra=True)
    assert report.ok
    assert "verified" in report.describe()


def test_truncation_is_caught_by_size_before_hashing(watchhand_root: Path) -> None:
    manifest = _manifest(watchhand_root)
    victim = watchhand_root / manifest.files[0].path
    victim.write_bytes(victim.read_bytes()[:-8])

    report = manifest.verify(watchhand_root, allow_extra=True)
    assert not report.ok
    assert manifest.files[0].path in report.size_mismatch


def test_same_size_different_bytes_is_caught(watchhand_root: Path) -> None:
    """The dangerous edit: a file whose size still matches.

    Nothing but a digest finds this, which is the whole argument for hashing.
    """
    manifest = _manifest(watchhand_root)
    target = next(r for r in manifest.files if r.path.endswith(".npy"))
    victim = watchhand_root / target.path
    payload = bytearray(victim.read_bytes())
    payload[-1] ^= 0xFF
    victim.write_bytes(bytes(payload))

    report = manifest.verify(watchhand_root, allow_extra=True)
    assert report.changed == (target.path,)
    assert report.size_mismatch == ()
    with pytest.raises(IntegrityError, match="content changed"):
        report.raise_if_bad()


def test_shallow_verify_misses_what_it_says_it_misses(watchhand_root: Path) -> None:
    manifest = _manifest(watchhand_root)
    target = next(r for r in manifest.files if r.path.endswith(".npy"))
    payload = bytearray((watchhand_root / target.path).read_bytes())
    payload[-1] ^= 0xFF
    (watchhand_root / target.path).write_bytes(bytes(payload))

    assert manifest.verify(watchhand_root, deep=False, allow_extra=True).ok
    assert not manifest.verify(watchhand_root, deep=True, allow_extra=True).ok


def test_missing_and_extra_files(watchhand_root: Path) -> None:
    manifest = _manifest(watchhand_root)
    (watchhand_root / manifest.files[0].path).unlink()
    (watchhand_root / "Study-1" / "stowaway.txt").write_text("hi", encoding="utf-8")

    report = manifest.verify(watchhand_root)
    assert manifest.files[0].path in report.missing
    assert "Study-1/stowaway.txt" in report.extra
    assert manifest.verify(watchhand_root, allow_extra=True).extra == ()


def test_round_trip_through_json(watchhand_root: Path, tmp_path: Path) -> None:
    manifest = build_manifest(
        watchhand_root,
        dataset="watchhand",
        version="test-1",
        identities=("study1/p02/x", "study1/p01/x"),
        include_suffixes=_SUFFIXES,
        notes="why",
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    restored = Manifest.read(path)

    assert restored.files == manifest.files
    assert restored.total_bytes == manifest.total_bytes
    assert restored.identities == ("study1/p01/x", "study1/p02/x"), "identities sorted"
    assert restored.notes == "why"


def test_unknown_schema_refused(tmp_path: Path) -> None:
    blob = {
        "schema": "something-else/9",
        "dataset": "watchhand",
        "version": "v1",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "files": [],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(IntegrityError, match="schema"):
        Manifest.read(path)


def test_manifest_requires_a_version() -> None:
    with pytest.raises(ValueError, match="version"):
        Manifest(
            dataset="watchhand",
            version="",
            created_utc="2026-01-01T00:00:00+00:00",
            files=(),
        )


def test_duplicate_paths_refused() -> None:
    record = FileRecord(path="a/b.npy", size_bytes=1, sha256="0" * 64)
    with pytest.raises(ValueError, match="duplicate"):
        Manifest(
            dataset="watchhand",
            version="v1",
            created_utc="2026-01-01T00:00:00+00:00",
            files=(record, record),
        )


@pytest.mark.parametrize(
    "path",
    ["/absolute/x.npy", "windows\\style.npy", ""],
)
def test_bad_record_paths_refused(path: str) -> None:
    with pytest.raises(ValueError):
        FileRecord(path=path, size_bytes=1, sha256="0" * 64)


def test_record_rejects_short_digest() -> None:
    with pytest.raises(ValueError, match="64 hex"):
        FileRecord(path="a.npy", size_bytes=1, sha256="deadbeef")


def test_unrecorded_file_is_refused_not_ignored(watchhand_root: Path) -> None:
    """A path absent from the manifest must raise, not fall through as fine."""
    manifest = _manifest(watchhand_root)
    with pytest.raises(IntegrityError, match="not in manifest"):
        manifest.verify_file(watchhand_root, "Study-1/nowhere.npy")


def test_lazy_verifier_hashes_each_file_once(watchhand_root: Path) -> None:
    manifest = _manifest(watchhand_root)
    verifier = LazyVerifier(manifest, watchhand_root)
    target = manifest.files[0].path

    assert verifier.check(target).is_file()
    assert verifier.verified_count == 1
    verifier.check(target)
    assert verifier.verified_count == 1


def test_lazy_verifier_raises_on_tampered_file(watchhand_root: Path) -> None:
    manifest = _manifest(watchhand_root)
    target = next(r for r in manifest.files if r.path.endswith(".json"))
    (watchhand_root / target.path).write_text("{}", encoding="utf-8")

    with pytest.raises(IntegrityError):
        LazyVerifier(manifest, watchhand_root).check(target.path)


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 8192
    path.write_bytes(payload)
    assert sha256_file(path, chunk_size=997) == hashlib.sha256(payload).hexdigest()


def test_missing_root_is_an_integrity_error(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError, match="does not exist"):
        build_manifest(tmp_path / "nope", dataset="watchhand", version="v1")


def test_schema_tag_is_written(watchhand_root: Path) -> None:
    assert _manifest(watchhand_root).to_json()["schema"] == MANIFEST_SCHEMA
