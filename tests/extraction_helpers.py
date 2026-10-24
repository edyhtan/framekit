"""Helpers for the extraction tests: example lens specs and input files. No footage."""

from pathlib import Path
from typing import Any

import yaml

PACE_LENS = {
    "lens": {
        "name": "pace",
        "version": "0.1.0",
        "description": "How fast a moment feels.",
        "requires_modules": ["timing"],
        "fields": {
            "tempo": {"type": "label", "vocabulary": ["slow", "steady", "fast"]},
            "cues": {"type": "labels", "vocabulary": ["pause", "rush", "hold"], "min": 1, "max": 2},
            "level": {"type": "integer", "min": 1, "max": 5},
        },
        "prompts": {
            "interpretation": {"file": "interpret.md"},
            "phrasing": "Searches a person would type.",
            "phrasing_style": "short, first person",
        },
        "validators": [{"kind": "required", "fields": ["tempo"]}],
        "consumer": {"judge_rubric": "Does the pace match?"},
    }
}

COLOUR_LENS = {
    "lens": {
        "name": "colour",
        "version": "1.0.0",
        "targets": ["scene"],
        "fields": {"palette": {"type": "label", "vocabulary": ["warm", "cool", "mixed"]}},
        "prompts": {"interpretation": "Say which palette dominates."},
    }
}


def write_yaml(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path
