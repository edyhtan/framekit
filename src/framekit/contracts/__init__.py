"""Contracts: provider-neutral ports and the types that cross them. Pure: no provider SDK,
no I/O library."""

from framekit.contracts.conversation import (
    STOP_REASONS,
    ChatMessage,
    ConversationPort,
    ConversationRequest,
    ConversationResult,
    ModelCallError,
    TextDelta,
)
from framekit.contracts.response_store import ResponseStore

__all__ = [
    "STOP_REASONS",
    "ChatMessage",
    "ConversationPort",
    "ConversationRequest",
    "ConversationResult",
    "ModelCallError",
    "ResponseStore",
    "TextDelta",
]
