"""Media work for extraction: ffmpeg (probe, analysis copy, decoding, encoding), the L1
signal detectors, and frames for models (read, pixel filters, images or storyboards).

Needs numpy (`pip install "framekit[media]"`) and an ffmpeg binary.
"""

from framekit.extraction.media.detectors import (
    DETECTOR_KINDS,
    Detector,
    MediaContext,
    detectors,
    get_detector,
    register_detector,
)
from framekit.extraction.media.ffmpeg import FFmpeg, MediaError, MediaInfo

__all__ = [
    "DETECTOR_KINDS",
    "Detector",
    "FFmpeg",
    "MediaContext",
    "MediaError",
    "MediaInfo",
    "detectors",
    "get_detector",
    "register_detector",
]
