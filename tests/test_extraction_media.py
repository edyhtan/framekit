"""Media work on synthetic video: probing, the analysis copy, signals, frames, pixel filters.

Videos are ffmpeg test patterns with beeps, generated per test (no footage). Tests that
need ffmpeg skip without it; set FRAMEKIT_TEST_FFMPEG to a binary's path to use one that
isn't on PATH.
"""

import asyncio
import os

import numpy as np
import pytest
from extraction_helpers import ffmpeg_binary, make_video, needs_ffmpeg

from framekit.extraction.media.detectors import (
    MediaContext,
    _onsets,
    get_detector,
    register_detector,
)
from framekit.extraction.media.ffmpeg import (
    FFmpeg,
    MediaError,
    copy_size,
    jpeg_scale,
    parse_probe,
    scaled_size,
)
from framekit.extraction.media.frames import _grid, deliver, read_frames
from framekit.extraction.schemas.plan import (
    AudioSignals,
    MotionSettings,
    NormalizeSettings,
    ShotSettings,
)
from framekit.vision import VisionProfile, resolve_profile
from framekit.vision.pixels import (
    PixelFrame,
    apply_pixel_filters,
    hamming,
    phash,
    sharpness,
)

PROBE = "\n".join(
    [
        "Input #0, matroska,webm, from 'x.mkv':",
        "  Duration: 00:07:01.53, start: 0.000000, bitrate: 2400 kb/s",
        "  Stream #0:0: Video: mjpeg (Baseline), yuvj420p, 300x300, 90k tbr (attached pic)",
        "  Stream #0:1(eng): Video: h264 (High), yuv420p(tv, bt709, progressive), 1440x1080 "
        "[SAR 4:3 DAR 16:9], 23.98 fps, 23.98 tbr, 1k tbn (default)",
        "      Side data:",
        "        displaymatrix: rotation of -90.00 degrees",
        "  Stream #0:2(eng): Audio: ac3, 48000 Hz, 5.1(side), fltp, 384 kb/s (default)",
    ]
)


def test_the_probe_reads_duration_streams_and_rotation():
    info = parse_probe(PROBE)
    assert info.duration_s == pytest.approx(421.53)
    assert (info.width, info.height) == (1080, 1440)  # cover art skipped, rotation applied
    assert info.fps == pytest.approx(23.98) and info.video_codec == "h264"
    assert info.has_audio and info.sample_rate_hz == 48000


def test_a_file_without_duration_or_video_is_refused():
    with pytest.raises(MediaError, match="no duration"):
        parse_probe("garbage")
    with pytest.raises(MediaError, match="no video stream"):
        parse_probe("  Duration: 00:00:05.00, start: 0\n  Stream #0:0: Audio: aac, 44100 Hz, mono")


def test_sizes_keep_the_aspect_and_never_upscale():
    info = parse_probe(PROBE.replace("rotation of -90.00", "rotation of 0.00"))
    assert copy_size(info, 720) == (960, 720)  # 1440x1080 shown at 16:9 is stored 4:3 here
    assert scaled_size(1280, 720, 768) == (768, 432)
    assert scaled_size(320, 180, 768) == (320, 180)
    assert jpeg_scale(100) == 2 and jpeg_scale(1) == 31 and jpeg_scale(85) == 6


def test_the_ffmpeg_binary_must_exist():
    with pytest.raises(MediaError, match="ffmpeg not found"):
        FFmpeg("definitely-not-ffmpeg-here")


# --- pixel filters (numpy only) --------------------------------------------------------


def gradient(shift: int = 0, size: int = 64) -> np.ndarray:
    x = np.arange(size)
    return ((x[None, :] * 4 + x[:, None] * 2 + shift) % 256).astype(np.uint8)


def test_phash_matches_near_copies_and_separates_different_frames():
    a, b = gradient(), gradient(2)
    noise = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    assert hamming(phash(a), phash(b)) <= 6
    assert hamming(phash(a), phash(noise)) > 10


def test_sharpness_prefers_detail():
    sharp = np.zeros((32, 32), np.uint8)
    sharp[::2, ::2] = 255
    assert sharpness(sharp) > sharpness(np.full((32, 32), 128, np.uint8))


def ref(name, **settings):
    from framekit.vision.params import StrategyRef

    return StrategyRef(name=name, version=1, params=settings)


def test_dedupe_drops_consecutive_copies_but_never_an_anchor():
    frames = [
        PixelFrame(0.0, gradient()),
        PixelFrame(0.5, gradient(1)),  # a near copy: dropped
        PixelFrame(1.0, gradient(1), anchor=True),  # a copy, but an anchor: kept
        PixelFrame(1.5, np.random.default_rng(1).integers(0, 255, (64, 64), dtype=np.uint8)),
    ]
    kept, notes = apply_pixel_filters([ref("dedupe", method="phash", max_distance=6)], frames)
    assert [f.time_s for f in kept] == [0.0, 1.0, 1.5]
    assert notes == ["dedupe@1 dropped 1 of 4 frames"]


def test_sharpness_keeps_the_best_frame_per_window_and_every_anchor():
    blur = np.full((32, 32), 100, np.uint8)
    detail = np.zeros((32, 32), np.uint8)
    detail[::2] = 255
    frames = [
        PixelFrame(0.1, blur),
        PixelFrame(0.4, detail),
        PixelFrame(0.6, blur, anchor=True),
        PixelFrame(1.2, blur),
    ]
    kept, _ = apply_pixel_filters([ref("sharpness", keep="best_per_window", window_s=1.0)], frames)
    assert [f.time_s for f in kept] == [0.4, 0.6, 1.2]


def test_a_storyboard_tiles_frames_and_leaves_empty_tiles_black():
    from framekit.extraction.media.frames import Frame

    tile = bytes([255, 0, 0]) * (4 * 2)
    frames = [Frame(0, 0, 0, (4, 2), tile, False)] * 3
    raw, size = _grid(frames, rows=2, cols=2)
    assert size == (8, 4)
    pixels = np.frombuffer(raw, np.uint8).reshape(4, 8, 3)
    assert pixels[0, 0].tolist() == [255, 0, 0] and pixels[3, 7].tolist() == [0, 0, 0]


def test_onsets_find_plucked_tones():
    rate = 16000
    t = np.arange(rate * 4) / rate
    signal = np.zeros_like(t)
    for start in (0.5, 1.7, 3.0):  # an instant attack, then a smooth decay: one onset each
        note = (t >= start) & (t < start + 0.3)
        signal[note] = 0.8 * np.sin(2 * np.pi * 880 * t[note]) * np.exp(-(t[note] - start) / 0.05)
    found = _onsets(signal, sensitivity=0.5)
    assert len(found) == 3
    assert all(abs(f - s) < 0.06 for f, s in zip(found, (0.5, 1.7, 3.0), strict=True))


def test_silence_has_no_onsets():
    assert _onsets(np.zeros(16000 * 2), sensitivity=1.0) == []


def test_detectors_are_named_and_versioned():
    assert get_detector("shots", "ffmpeg_scene").key == "ffmpeg_scene@1"
    with pytest.raises(LookupError, match="known: ffmpeg_scene"):
        get_detector("shots", "transnet")
    with pytest.raises(ValueError, match="already registered"):
        register_detector(get_detector("motion", "frame_difference"))


# --- with ffmpeg -----------------------------------------------------------------------


@pytest.fixture
def ffmpeg():
    binary = ffmpeg_binary()
    if binary is None:
        if os.environ.get("FRAMEKIT_REQUIRE_FFMPEG") == "1":
            pytest.fail("ffmpeg is required (FRAMEKIT_REQUIRE_FFMPEG=1) but isn't installed")
        pytest.skip("ffmpeg isn't installed")
    return FFmpeg(binary)


@needs_ffmpeg
def test_normalize_then_detect_cuts_onsets_and_motion(tmp_path, ffmpeg):
    source = make_video(tmp_path / "source.mp4", size=(640, 360))

    async def run():
        info = await ffmpeg.probe(source)
        copy = await ffmpeg.normalize(
            source, tmp_path / "copy.mp4", NormalizeSettings(fps=8, height_px=240), info
        )
        context = MediaContext(tmp_path / "copy.mp4", copy, ffmpeg)
        cuts = await get_detector("shots", "ffmpeg_scene").run(ShotSettings(), context)
        audio = await get_detector("audio", "spectral_flux").run(AudioSignals(), context)
        motion = await get_detector("motion", "frame_difference").run(MotionSettings(), context)
        return info, copy, cuts, audio, motion

    info, copy, cuts, audio, motion = asyncio.run(run())
    assert info.duration_s == pytest.approx(24, abs=0.1) and info.has_audio
    assert (copy.width, copy.height, copy.fps) == (426, 240, 8)
    assert source.exists()  # the source is only read
    assert [round(c) for c in cuts] == [8, 16]
    onsets = audio["onsets"]
    for beep in (2.0, 5.0, 11.0, 19.0):
        assert any(abs(o - beep) < 0.1 for o in onsets), (beep, onsets)
    assert audio["loudness"] and len(motion) == 24


@needs_ffmpeg
def test_frames_are_read_at_the_profile_size_and_delivered(tmp_path, ffmpeg):
    source = make_video(tmp_path / "source.mp4", audio=False, duration=10, cuts=(5.0,))
    profile = resolve_profile(
        VisionProfile.from_mapping(
            {
                "version": 1,
                "frame_selection": {"candidates": {"strategy": "uniform", "fps": 1}},
                "image": {"max_side_px": 160, "format": "jpeg", "quality": 80},
                "delivery": {"mode": "storyboard", "storyboard": {"rows": 1, "cols": 2}},
            }
        )
    )

    async def run():
        info = await ffmpeg.probe(source)
        limit = asyncio.Semaphore(2)
        frames, notes = await read_frames(
            ffmpeg, source, info, [1.0, 1.01, 4.99, 5.0], {5.0}, 160, limit
        )
        images = await deliver(ffmpeg, profile, frames, limit)
        return frames, notes, images

    frames, notes, images = asyncio.run(run())
    assert [f.planned_s for f in frames] == [1.0, 4.99, 5.0]  # 1.01 is the frame of 1.0
    assert "fell on frames already chosen" in notes[0]
    assert frames[0].size == (160, 90) and frames[-1].anchor
    assert frames[1].gray().mean() != frames[2].gray().mean()  # either side of the cut
    assert len(images) == 2 and images[0].size == (320, 90)
    assert (
        images[0].image.data[:3] == b"\xff\xd8\xff" and images[0].image.media_type == "image/jpeg"
    )
