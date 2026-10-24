"""Cache keys and stored entries.

The key hashes everything that shapes a reply: the resolved model, the system
prompt, the history, the message, the budget, the reasoning switch and the
response schema. Any change to a prompt, the code's request or a setting is a miss, so a
stale reply is never reused.
"""

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal
from typing import Any

from framekit.contracts.conversation import ConversationRequest, ConversationResult, TextDelta

KEY_PREFIX = "llmcache"


def request_key(
    request: ConversationRequest, model_id: str, namespace: str, method: str = "generate"
) -> str:
    """`method` is part of the key: a streamed reply is replayed as a stream, a generated
    one as a single result."""
    shape = {
        "method": method,
        "model_id": model_id,
        "system": request.system,
        "history": [[m.role, m.content] for m in request.history],
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "reasoning": request.reasoning,
        "response_schema": request.response_schema,
    }
    canonical = json.dumps(shape, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"{KEY_PREFIX}:{namespace}:{request.cache_class or 'default'}:{digest}"


@dataclass
class Entry:
    """A recorded reply: the final result and the streamed pieces, with their timing
    (milliseconds after the request) for paced replays."""

    result: ConversationResult
    deltas: list[tuple[str, str, int]] = field(default_factory=list)  # (kind, text, at_ms)

    def encode(self) -> bytes:
        data = {
            "result": {
                k: str(v) if isinstance(v, Decimal) else v for k, v in asdict(self.result).items()
            },
            "deltas": self.deltas,
        }
        return zlib.compress(json.dumps(data, ensure_ascii=False).encode(), level=6)

    @classmethod
    def decode(cls, blob: bytes) -> "Entry":
        data = json.loads(zlib.decompress(blob))
        known = {f.name for f in fields(ConversationResult)}
        values: dict[str, Any] = {k: v for k, v in data["result"].items() if k in known}
        values["cost_usd"] = Decimal(values.get("cost_usd") or "0")
        return cls(ConversationResult(**values), [tuple(d) for d in data["deltas"]])

    def deltas_as_pieces(self) -> list[TextDelta]:
        return [TextDelta(kind, text) for kind, text, _ in self.deltas]  # type: ignore[arg-type]
