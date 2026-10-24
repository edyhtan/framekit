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


def base_plan() -> dict[str, Any]:
    """A small, fast plan on the scripted models."""
    return {
        "version": 1,
        "normalize": {"fps": 8, "height_px": 240},
        "vision_profiles": {
            "overview": {
                "version": 1,
                "frame_selection": {
                    "candidates": {"strategy": "uniform", "fps": 0.5},
                    "budget": {"max_frames_per_scene": 9},
                },
                "image": {"max_side_px": 160},
                "delivery": {"mode": "storyboard", "storyboard": {"rows": 3, "cols": 3}},
            },
            "detail": {
                "version": 1,
                "frame_selection": {
                    "candidates": {"strategy": "uniform", "fps": 2},
                    "anchors": ["shot_boundaries"],
                    "filters": [{"dedupe": {"max_distance": 4}}, {"sharpness": {"window_s": 1.0}}],
                    "budget": {"max_frames_per_scene": 4, "allocate_by": ["motion"]},
                },
                "image": {"max_side_px": 200, "format": "jpeg", "quality": 80},
            },
        },
        "stages": {"L2": {"vision": "overview"}, "L3a": {"vision": "detail"}},
        "segmentation": {"scene_length_s": {"min": 4, "target": 8, "max": 12}},
        "lens_passes": {"phrasings": {"count": 3}},
        "models": {
            "L2": {"model_id": "seer", "max_tokens": 1024},
            "L3a": {"model_id": "seer", "max_tokens": 1024},
            "L3b": {"model_id": "writer", "max_tokens": 512},
            "embed_text": {"model_id": "vectors"},
        },
        "governance": {"model_inputs": "local_only"},
    }
