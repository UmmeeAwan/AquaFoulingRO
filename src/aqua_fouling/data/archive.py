"""Safe ZIP extraction with path traversal rejection and SHA-256 checksums."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


class UnsafeArchiveError(ValueError):
    """Raised when an archive member path is unsafe."""


@dataclass(frozen=True)
class ExtractedFileRecord:
    """Checksum record for one extracted member."""

    archive_member: str
    destination: Path
    sha256: str
    size_bytes: int


@dataclass
class SafeExtractResult:
    """Result of a safe archive extraction."""

    archive_path: Path
    archive_sha256: str
    destination_root: Path
    files: list[ExtractedFileRecord] = field(default_factory=list)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_unsafe_member(member_name: str, destination_root: Path) -> str | None:
    """Return a reason string if the member path is unsafe, else None."""
    if not member_name or member_name.strip() == "":
        return "empty member name"
    # Zip members use forward slashes; reject absolute and parent traversal.
    normalised = member_name.replace("\\", "/")
    if normalised.startswith("/") or normalised.startswith("//"):
        return f"absolute path: {member_name}"
    parts = Path(normalised).parts
    if any(part == ".." for part in parts):
        return f"path traversal (..): {member_name}"
    # Drive letters / Windows absolute (e.g. C:/...)
    if Path(normalised).is_absolute():
        return f"absolute path: {member_name}"
    target = (destination_root / normalised).resolve()
    try:
        target.relative_to(destination_root.resolve())
    except ValueError:
        return f"escapes destination root: {member_name}"
    return None


def safe_extract_zip(
    archive_path: Path,
    destination_root: Path,
    *,
    overwrite: bool = True,
) -> SafeExtractResult:
    """Extract a ZIP archive safely, rejecting unsafe paths.

    The original archive is never modified. Only members that pass path checks
    are written under ``destination_root``.
    """
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        msg = f"Archive not found: {archive_path}"
        raise FileNotFoundError(msg)

    destination_root = destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    archive_digest = sha256_file(archive_path)
    records: list[ExtractedFileRecord] = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            # Skip directory entries after validating their names.
            reason = _is_unsafe_member(info.filename, destination_root)
            if reason is not None:
                msg = f"Rejected unsafe archive member ({reason})"
                raise UnsafeArchiveError(msg)

            if info.is_dir() or info.filename.endswith("/"):
                dir_path = destination_root / info.filename.replace("\\", "/")
                dir_path.mkdir(parents=True, exist_ok=True)
                continue

            target = destination_root / info.filename.replace("\\", "/")
            if target.exists() and not overwrite:
                msg = f"Refusing to overwrite existing file: {target}"
                raise FileExistsError(msg)
            target.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(info, "r") as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

            records.append(
                ExtractedFileRecord(
                    archive_member=info.filename,
                    destination=target,
                    sha256=sha256_file(target),
                    size_bytes=target.stat().st_size,
                )
            )

    return SafeExtractResult(
        archive_path=archive_path,
        archive_sha256=archive_digest,
        destination_root=destination_root,
        files=records,
    )
