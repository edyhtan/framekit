"""A `ResponseStore` on the local file system: one file per entry, for standalone runs,
benchmarks and tests.

Keys are hashed into file names (keys may contain characters a file system rejects).
Each file holds an 8-byte expiry time (Unix seconds) followed by the value. Writes are
atomic (a temporary file, then a rename), so a crash never leaves a half entry.
"""

import hashlib
import logging
import os
import struct
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)
_HEADER = struct.Struct(">d")  # expiry, Unix seconds


class FileResponseStore:
    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._dir / digest[:2] / digest

    async def get(self, key: str) -> bytes | None:
        path = self._path(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            log.warning("model cache read failed", exc_info=True)
            return None
        if len(data) < _HEADER.size:
            return None
        (expires_at,) = _HEADER.unpack_from(data)
        if expires_at < time.time():
            path.unlink(missing_ok=True)
            return None
        return data[_HEADER.size :]

    async def put(self, key: str, value: bytes, ttl_s: int) -> bool:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(_HEADER.pack(time.time() + ttl_s))
                    out.write(value)
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            return True
        except OSError:
            log.warning("model cache write failed", exc_info=True)
            return False

    def clear(self) -> int:
        """Delete every entry; returns how many were removed."""
        removed = 0
        for path in self._dir.glob("*/*"):
            if path.is_file() and not path.name.startswith(".tmp-"):
                path.unlink(missing_ok=True)
                removed += 1
        return removed
