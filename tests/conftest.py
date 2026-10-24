"""Shared fixtures."""

import pytest

from framekit.vision import registry


@pytest.fixture
def isolated_registry(monkeypatch):
    """Strategies a test registers are gone after it, so tests don't depend on their order
    and can run twice in one process."""
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
