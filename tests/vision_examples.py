"""Vision parameter examples shared by the vision tests, as plain data."""

import copy

# docs/settings.md §5.2, the vision part of plan.yaml, verbatim (YAML → Python).
SETTINGS_EXAMPLE = {
    "vision_profiles": {
        "scene_overview": {
            "version": 1,
            "frame_selection": {"candidates": {"strategy": "uniform", "fps": 1}},
            "image": {"max_side_px": 512, "aspect": "preserve"},
            "delivery": {"mode": "native_video"},
        },
        "scene_detail": {
            "version": 1,
            "views": {
                "mode": "overview_plus_detail",
                "overview": {"fps": 1, "max_frames": 6},
                "detail": {
                    "around": ["motion", "audio_onsets"],
                    "span_s": 3,
                    "max_views": 2,
                    "max_frames": 5,
                },
            },
            "frame_selection": {
                "candidates": {"strategy": "uniform", "fps": 4},
                "anchors": ["shot_boundaries"],
                "filters": [
                    {"dedupe": {"method": "phash", "max_distance": 6}},
                    {"sharpness": {"keep": "best_per_window", "window_s": 0.5}},
                ],
                "budget": {
                    "max_frames_per_scene": 16,
                    "allocate_by": ["motion", "audio_onsets"],
                    "min_gap_s": 0.25,
                },
            },
            "image": {"max_side_px": 768, "aspect": "preserve", "format": "jpeg", "quality": 85},
            "delivery": {"mode": "images"},
        },
        "keyframes": {
            "version": 1,
            "frame_selection": {
                "candidates": {"strategy": "shot_based", "pick": "middle"},
                "filters": [{"sharpness": {"keep": "best_per_window", "window_s": 0.5}}],
                "budget": {"max_frames_per_scene": 4},
            },
            "image": {"max_side_px": 384, "aspect": "preserve", "format": "jpeg", "quality": 85},
        },
    },
    "stages": {
        "L2": {"vision": "scene_overview"},
        "L3a": {"vision": "scene_detail"},
        "embed_image": {"vision": "keyframes"},
    },
}

# DESIGN.md §2.3 before 2026-10-23 22:31: dedupe, a frame budget and JPEG settings with
# native_video, which breaks the section's own rule.
OLD_NATIVE_VIDEO_EXAMPLE = {
    "vision_profiles": {
        "scene_overview": {
            "version": 1,
            "frame_selection": {
                "candidates": {"strategy": "uniform", "fps": 1},
                "anchors": [],
                "filters": [{"dedupe": {"method": "phash", "max_distance": 6}}],
                "budget": {"max_frames": 600},
            },
            "image": {"max_side_px": 512, "aspect": "preserve", "format": "jpeg", "quality": 80},
            "delivery": {"mode": "native_video"},
        }
    },
    "stages": {"L2": {"vision": "scene_overview"}},
}


def settings_example():
    return copy.deepcopy(SETTINGS_EXAMPLE)


def profile(**overrides):
    """A minimal valid images profile; keyword arguments replace top-level parts."""
    base = {
        "version": 1,
        "frame_selection": {"candidates": {"strategy": "uniform", "fps": 2}},
        "image": {"max_side_px": 512},
    }
    base.update(overrides)
    return base


def params_with(**profiles):
    return {
        "vision_profiles": profiles,
        "stages": {f"stage_{name}": {"vision": name} for name in profiles},
    }
