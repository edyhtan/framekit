"""A `ResponseStore` on the local file system: one file per entry, for standalone runs,
benchmarks and tests.

Layout: `<dir>/<namespace>/<cache class>/<xx>/<sha256 of the key>`, taken from the
cache's keys (`llmcache:<namespace>:<class>:...`), so one class can be counted or wiped
(`framekit cache stats|wipe`); other keys go under `_other/`. Key hashes are the file
names (keys may contain characters a file system rejects). Each file holds an 8-byte
expiry time (Unix seconds) followed by the value. Writes are atomic (a temporary file, then
a rename), so a crash never leaves a half entry.
"""

import hashlib
import logging
import os
import re
import struct
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)
_HEADER = struct.Struct(">d")  # expiry, Unix seconds
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")  # no dots: Windows drops trailing ones
_HEX = re.compile(r"^[0-9a-f]{64}$")
OTHER = "_other"


def _folder_name(text: str) -> str:
    return _UNSAFE.sub("_", text) or "_"


def key_parts(key: str) -> tuple[str, str]:
    """(namespace, cache class) of a response-cache key, or ("_other", "_other")."""
    parts = key.split(":", 3)
    if len(parts) == 4 and parts[0] == "llmcache":
        return _folder_name(parts[1]), _folder_name(parts[2])
    return OTHER, OTHER


class FileResponseStore:
    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    @property
    def directory(self) -> Path:
        return self._dir

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        namespace, cache_class = key_parts(key)
        if namespace == OTHER:
            return self._dir / OTHER / digest[:2] / digest
        return self._dir / namespace / cache_class / digest[:2] / digest

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

    def _entries(self) -> list[tuple[str, str, Path]]:
        """(namespace, class, file) of every entry, the older flat layout included."""
        found = []
        for path in self._dir.rglob("*"):
            if not _HEX.match(path.name) or not path.is_file():
                continue
            parts = path.relative_to(self._dir).parts
            if len(parts) == 4:
                found.append((parts[0], parts[1], path))
            elif len(parts) == 3 and parts[0] == OTHER:
                found.append((OTHER, OTHER, path))
            elif len(parts) == 2:  # the layout before classes had folders
                found.append(("_flat", "_flat", path))
        return found

    def stats(self) -> dict[str, dict[str, int]]:
        """Entries, bytes and expired entries per `namespace/class`."""
        result: dict[str, dict[str, int]] = {}
        moment = time.time()
        for namespace, cache_class, path in self._entries():
            try:
                size = path.stat().st_size
                with path.open("rb") as handle:
                    header = handle.read(_HEADER.size)
            except OSError:
                continue
            row = result.setdefault(
                f"{namespace}/{cache_class}", {"entries": 0, "bytes": 0, "expired": 0}
            )
            row["entries"] += 1
            row["bytes"] += size
            if len(header) == _HEADER.size and _HEADER.unpack(header)[0] < moment:
                row["expired"] += 1
        return dict(sorted(result.items()))

    def clear(self, cache_class: str | None = None, namespace: str | None = None) -> int:
        """Delete entries (all, or those of one class and/or namespace); returns how many."""
        removed = 0
        for entry_namespace, entry_class, path in self._entries():
            if cache_class is not None and entry_class != _folder_name(cache_class):
                continue
            if namespace is not None and entry_namespace != _folder_name(namespace):
                continue
            path.unlink(missing_ok=True)
            removed += 1
        return removed
