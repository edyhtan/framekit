"""Contracts: provider-neutral ports and the types that cross them. Pure: no provider SDK,
no I/O library."""

from framekit.contracts.conversation import (
    STOP_REASONS,
    ChatMessage,
    ConversationPort,
    ConversationRequest,
    ConversationResult,
    ImagePart,
    ModelCallError,
    TextDelta,
)
from framekit.contracts.embedding import EmbeddingPort, EmbeddingRequest, EmbeddingResult
from framekit.contracts.response_store import ResponseStore

__all__ = [
    "STOP_REASONS",
    "ChatMessage",
    "ConversationPort",
    "ConversationRequest",
    "ConversationResult",
    "EmbeddingPort",
    "EmbeddingRequest",
    "EmbeddingResult",
    "ImagePart",
    "ModelCallError",
    "ResponseStore",
    "TextDelta",
]
