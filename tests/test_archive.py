"""Safe archive extraction tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aqua_fouling.data.archive import UnsafeArchiveError, safe_extract_zip, sha256_file


def test_safe_extract_writes_files_and_checksums(fixtures_dir: Path, tmp_path: Path) -> None:
    archive = fixtures_dir / "good_archive.zip"
    dest = tmp_path / "out"
    result = safe_extract_zip(archive, dest)

    assert result.archive_sha256 == sha256_file(archive)
    assert len(result.files) >= 2
    hello = dest / "safe_archive" / "hello.txt"
    assert hello.is_file()
    assert hello.read_text(encoding="utf-8") == "hello"
    record = next(f for f in result.files if f.destination == hello)
    assert record.sha256 == sha256_file(hello)


def test_unsafe_archive_rejected(fixtures_dir: Path, tmp_path: Path) -> None:
    archive = fixtures_dir / "unsafe_archive.zip"
    with pytest.raises(UnsafeArchiveError, match="path traversal"):
        safe_extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
