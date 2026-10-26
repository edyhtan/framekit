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


def write_inputs(
    root: Path,
    video: Path,
    *,
    lenses: tuple[str, ...] = ("pace",),
    plan: dict[str, Any] | None = None,
    space: dict[str, Any] | None = None,
    rights_mode: str = "byo_media",
) -> dict[str, Path]:
    """Lens specs, a search space, a corpus and a plan as files, like a user's."""
    write_yaml(root / "lenses" / "pace" / "lens.yaml", PACE_LENS)
    (root / "lenses" / "pace" / "interpret.md").write_text(
        "Rate how fast the moment feels.", encoding="utf-8"
    )
    write_yaml(root / "lenses" / "colour.yaml", COLOUR_LENS)
    refs = {
        "pace": {"spec": "lenses/pace", "version": "0.1.0"},
        "colour": {"spec": "lenses/colour.yaml", "version": "1.0.0"},
    }
    space_data = {
        "search_space": {
            "version": 1,
            "intent": "test patterns",
            "lenses": [refs[name] for name in lenses],
            "resolution": {"moment_length_s": {"min": 3, "max": 6}},
            "query_style": {"examples": ["when everything moves at once"]},
            **(space or {}),
        }
    }
    corpus_data = {
        "corpus": {
            "profile": {
                "name": "synthetic",
                "unit_definition": "one generated test video per file",
                "metadata_fields": {"title": "str", "quality": ["good", "fair"]},
                "entities": ["Ada", "Brook"],
                "perception_hints": ["generated test patterns"],
                "rights_mode": rights_mode,
            },
            "manifest": [
                {"unit_id": "u1", "expected_duration_s": 24.0, "excluded_ranges": [[0.0, 1.0]]}
            ],
            "units": [
                {"unit_id": "u1", "path": str(video), "title": "Patterns", "quality": "good"}
            ],
        }
    }
    plan_data = {"plan": merged(base_plan(), plan or {})}
    return {
        "space": write_yaml(root / "search-space.yaml", space_data),
        "corpus": write_yaml(root / "corpus.yaml", corpus_data),
        "plan": write_yaml(root / "plan.yaml", plan_data),
    }


def merged(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """`base` with `overrides` in it: mappings merge key by key, anything else replaces."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merged(out[key], value)
        else:
            out[key] = value
    return out


def base_plan() -> dict[str, Any]:
    """A small, fast plan on the scripted models."""
    return {
        "version": 2,
        "observation": {
            "normalize": {"fps": 8, "height_px": 240},
            "frames": {
                "profiles": {
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
                            "filters": [
                                {"dedupe": {"max_distance": 4}},
                                {"sharpness": {"window_s": 1.0}},
                            ],
                            "budget": {"max_frames_per_scene": 4, "allocate_by": ["motion"]},
                        },
                        "image": {"max_side_px": 200, "format": "jpeg", "quality": 80},
                    },
                },
                "stages": {"L2": {"profile": "overview"}, "L3a": {"profile": "detail"}},
            },
        },
        "structure": {"segmentation": {"scene_length_s": {"min": 4, "target": 8, "max": 12}}},
        "interpretation": {"lens_passes": {"phrasings": {"count": 3}}},
        "models": {
            "L2": {"model_id": "seer", "max_tokens": 1024},
            "L3a": {"model_id": "seer", "max_tokens": 1024},
            "L3b": {"model_id": "writer", "max_tokens": 512},
            "embed_text": {"model_id": "vectors"},
        },
        "policy": {"governance": {"model_inputs": "local_only"}},
    }
