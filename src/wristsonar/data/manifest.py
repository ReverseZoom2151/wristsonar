"""Content-addressed integrity for a dataset that lives outside the repository.

WatchHand is roughly 175 GB downloaded as twelve zip archives from a
repository that can reissue them. Nothing about that path is tamper-evident:
an unzip that silently truncated, a partial re-download, a well-meaning
colleague who deleted the sessions that "looked wrong", or a version bump
upstream all produce a directory tree that still loads and still trains and
still reports a number. The number is then attached to data nobody can
identify afterwards.

So a manifest here is not a convenience index. It is the thing that makes an
evaluation citable. `Protocol` in `wristsonar.protocol` already refuses to
exist without a `dataset_version` string; this module is what makes that
string mean something, by pinning it to the digests of the bytes it names.

Two verification modes exist because the dataset is large. `verify` walks
everything and is what you run once after downloading and again before
publishing a number. `LazyVerifier` checks a file's digest at the moment the
loader opens it, which costs one hash of one session and catches the case
that actually happens in practice: the file you are reading right now is not
the file you recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

__all__ = [
    "MANIFEST_SCHEMA",
    "FileRecord",
    "IntegrityError",
    "LazyVerifier",
    "Manifest",
    "VerificationReport",
    "build_manifest",
    "sha256_file",
]

MANIFEST_SCHEMA = "wristsonar.dataset-manifest/1"
"""Schema tag written into every manifest.

Versioned separately from the dataset so that a future change to what we
record cannot be mistaken for a change to what was recorded.
"""

_HASH_CHUNK = 1 << 20
"""One mebibyte. Large enough that syscall overhead vanishes, small enough
that hashing a 62 MB echo profile does not fault a process out of memory."""


class IntegrityError(RuntimeError):
    """Raised when the bytes on disk are not the bytes the manifest names.

    Deliberately not a warning and deliberately not recoverable by default.
    The failure mode this module exists to prevent is an evaluation that
    completes successfully against the wrong data, so the only safe response
    is to stop.
    """


def sha256_file(path: Path, *, chunk_size: int = _HASH_CHUNK) -> str:
    """Hex digest of a file, streamed.

    SHA-256 rather than a fast non-cryptographic hash because a manifest is a
    provenance claim that may be read by someone who does not trust the person
    who wrote it.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One file's identity: where it sits, how big it is, what it hashes to.

    `size_bytes` is redundant against the digest but is checked first, because
    it is free and it turns the overwhelmingly common failure (a truncated
    download) into an instant, legible error rather than a minute of hashing.
    """

    path: str
    """Slash-separated path relative to the dataset root, on every platform."""

    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("a file record must name a path")
        # PurePosixPath treats a backslash as an ordinary character, so a
        # Windows-style path would round-trip unnoticed and then fail to
        # resolve on Linux. Reject it explicitly.
        if "\\" in self.path or self.path != PurePosixPath(self.path).as_posix():
            raise ValueError(
                "manifest paths must be relative and slash-separated, "
                f"got {self.path!r}"
            )
        if PurePosixPath(self.path).is_absolute() or self.path.startswith("/"):
            raise ValueError(f"manifest paths must be relative, got {self.path!r}")
        if self.size_bytes < 0:
            raise ValueError("size_bytes cannot be negative")
        if len(self.sha256) != 64:
            raise ValueError(
                f"sha256 must be 64 hex characters, got {len(self.sha256)}"
            )

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> FileRecord:
        return cls(
            path=str(blob["path"]),
            size_bytes=int(blob["size_bytes"]),
            sha256=str(blob["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """What a verification pass found, in full, before anyone raises.

    Reporting every discrepancy rather than the first one matters because the
    difference between "one session is corrupt" and "the whole tree is a
    different release" is the difference between re-downloading a file and
    discarding a set of results.
    """

    missing: tuple[str, ...] = ()
    """Recorded in the manifest, absent on disk."""

    extra: tuple[str, ...] = ()
    """Present on disk, absent from the manifest."""

    size_mismatch: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    """Right size, wrong digest. The dangerous one, because it is invisible."""

    @property
    def ok(self) -> bool:
        return not (self.missing or self.extra or self.size_mismatch or self.changed)

    def describe(self) -> str:
        if self.ok:
            return "manifest verified: all files present and unchanged"
        parts: list[str] = []
        for label, paths in (
            ("missing", self.missing),
            ("unrecorded", self.extra),
            ("wrong size", self.size_mismatch),
            ("content changed", self.changed),
        ):
            if paths:
                shown = ", ".join(paths[:5])
                more = f" (+{len(paths) - 5} more)" if len(paths) > 5 else ""
                parts.append(f"{len(paths)} {label}: {shown}{more}")
        return "; ".join(parts)

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise IntegrityError(self.describe())


@dataclass(frozen=True, slots=True)
class Manifest:
    """A dataset release, pinned.

    `identities` carries the participant, session and device labels that the
    loader parsed out of the directory tree. Recording them alongside the
    digests means a manifest answers "who is in this evaluation" without
    mounting 175 GB, and means a tree that has the right bytes in the wrong
    places is still caught.
    """

    dataset: str
    version: str
    created_utc: str
    files: tuple[FileRecord, ...]
    identities: tuple[str, ...] = ()
    """Sorted, opaque session identity strings, one per session."""

    notes: str = ""
    schema: str = MANIFEST_SCHEMA

    _by_path: dict[str, FileRecord] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("a manifest must name its dataset")
        if not self.version:
            raise ValueError(
                "a manifest must pin a version; an unversioned dataset "
                "cannot be reproduced"
            )
        index: dict[str, FileRecord] = {}
        for record in self.files:
            if record.path in index:
                raise ValueError(f"duplicate path in manifest: {record.path}")
            index[record.path] = record
        object.__setattr__(self, "_by_path", index)

    def __len__(self) -> int:
        return len(self.files)

    def __contains__(self, path: str) -> bool:
        return path in self._by_path

    @property
    def total_bytes(self) -> int:
        return sum(record.size_bytes for record in self.files)

    def record(self, relative_path: str) -> FileRecord:
        try:
            return self._by_path[relative_path]
        except KeyError as exc:
            raise IntegrityError(
                f"{relative_path} is not in manifest {self.dataset}@{self.version}; "
                "refusing to read a file the manifest does not account for"
            ) from exc

    def verify_file(self, root: Path, relative_path: str) -> None:
        """Check one file against its record, or raise.

        The single-file entry point exists so that a loader can pay for
        integrity only on the sessions it actually touches.
        """
        expected = self.record(relative_path)
        actual = root / PurePosixPath(relative_path)
        if not actual.is_file():
            raise IntegrityError(f"{relative_path} is recorded but missing from {root}")
        size = actual.stat().st_size
        if size != expected.size_bytes:
            raise IntegrityError(
                f"{relative_path} is {size} bytes, manifest says "
                f"{expected.size_bytes}; the file is truncated or replaced"
            )
        digest = sha256_file(actual)
        if digest != expected.sha256:
            raise IntegrityError(
                f"{relative_path} hashes to {digest[:16]}..., manifest says "
                f"{expected.sha256[:16]}...; contents have changed"
            )

    def verify(
        self,
        root: Path,
        *,
        deep: bool = True,
        allow_extra: bool = False,
    ) -> VerificationReport:
        """Walk the whole tree and compare it to what was recorded.

        `deep=False` compares only presence and size. That is not integrity
        and the docstring says so plainly, but it turns a multi-hour hash of
        175 GB into seconds, which is the difference between a check that runs
        on every CI job and one that never runs at all.
        """
        root = Path(root)
        if not root.is_dir():
            raise IntegrityError(f"dataset root {root} does not exist")

        on_disk = {p.as_posix(): root / p for p in _walk_relative(root)}
        missing: list[str] = []
        size_mismatch: list[str] = []
        changed: list[str] = []

        for record in self.files:
            actual = on_disk.pop(record.path, None)
            if actual is None:
                missing.append(record.path)
                continue
            if actual.stat().st_size != record.size_bytes:
                size_mismatch.append(record.path)
                continue
            if deep and sha256_file(actual) != record.sha256:
                changed.append(record.path)

        extra = () if allow_extra else tuple(sorted(on_disk))
        return VerificationReport(
            missing=tuple(sorted(missing)),
            extra=extra,
            size_mismatch=tuple(sorted(size_mismatch)),
            changed=tuple(sorted(changed)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dataset": self.dataset,
            "version": self.version,
            "created_utc": self.created_utc,
            "notes": self.notes,
            "identities": list(self.identities),
            "files": [record.to_json() for record in self.files],
        }

    def write(self, path: Path) -> None:
        """Write sorted, indented JSON so that two manifests diff usefully."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> Manifest:
        schema = str(blob.get("schema", ""))
        if schema != MANIFEST_SCHEMA:
            raise IntegrityError(
                f"manifest schema {schema!r} is not {MANIFEST_SCHEMA!r}; "
                "refusing to guess at an unknown format"
            )
        return cls(
            dataset=str(blob["dataset"]),
            version=str(blob["version"]),
            created_utc=str(blob["created_utc"]),
            files=tuple(FileRecord.from_json(f) for f in blob["files"]),
            identities=tuple(str(i) for i in blob.get("identities", ())),
            notes=str(blob.get("notes", "")),
            schema=schema,
        )

    @classmethod
    def read(cls, path: Path) -> Manifest:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def build_manifest(
    root: Path,
    *,
    dataset: str,
    version: str,
    identities: Iterable[str] = (),
    include_suffixes: Sequence[str] | None = None,
    notes: str = "",
) -> Manifest:
    """Hash a dataset tree into a manifest.

    `include_suffixes` exists because the shipped tree carries incidental
    files, .DS_Store and per-session PNG previews among them, whose presence
    or absence says nothing about the data. Recording them would make the
    manifest fail for reasons that do not matter, and a check that fails for
    reasons that do not matter gets disabled.
    """
    root = Path(root)
    if not root.is_dir():
        raise IntegrityError(f"dataset root {root} does not exist")
    wanted = None if include_suffixes is None else {s.lower() for s in include_suffixes}

    records: list[FileRecord] = []
    for relative in _walk_relative(root):
        if wanted is not None and relative.suffix.lower() not in wanted:
            continue
        absolute = root / relative
        records.append(
            FileRecord(
                path=relative.as_posix(),
                size_bytes=absolute.stat().st_size,
                sha256=sha256_file(absolute),
            )
        )
    records.sort(key=lambda r: r.path)

    return Manifest(
        dataset=dataset,
        version=version,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        files=tuple(records),
        identities=tuple(sorted(identities)),
        notes=notes,
    )


class LazyVerifier:
    """Verifies files on first read and remembers what it has already seen.

    A loader that re-hashed a 62 MB profile on every access would be unusable,
    and a loader that verified nothing would be pointless. Caching by relative
    path is safe here because the manifest is immutable and the dataset is
    read-only by construction.
    """

    __slots__ = ("_manifest", "_root", "_seen")

    def __init__(self, manifest: Manifest, root: Path) -> None:
        self._manifest = manifest
        self._root = Path(root)
        self._seen: set[str] = set()

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    @property
    def root(self) -> Path:
        return self._root

    @property
    def verified_count(self) -> int:
        return len(self._seen)

    def check(self, relative_path: str) -> Path:
        """Verify if not already verified, and return the absolute path."""
        if relative_path not in self._seen:
            self._manifest.verify_file(self._root, relative_path)
            self._seen.add(relative_path)
        return self._root / PurePosixPath(relative_path)


def _walk_relative(root: Path) -> Iterator[PurePosixPath]:
    """Every file under root, as a sorted relative posix path.

    os.walk rather than Path.rglob because the sort has to be applied at each
    directory level to make the traversal order stable across filesystems that
    disagree about directory ordering.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        base = Path(dirpath)
        for name in sorted(filenames):
            yield PurePosixPath((base / name).relative_to(root).as_posix())
