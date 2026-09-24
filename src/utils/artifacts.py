"""Small helpers for immutable pipeline artifacts."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: str | Path, writer: Callable[[Path], None]) -> Path:
    """Write a new file through a same-directory temporary path."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing output: {destination}")
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    writer(temporary)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"writer did not create a nonempty file: {temporary}")
    os.rename(temporary, destination)
    return destination
