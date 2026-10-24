"""Synthetic media for the tests: ffmpeg test patterns and tones generated per test (no
footage), and the ffmpeg to use."""

import os
import shutil
import subprocess  # noqa: S404  (tests drive a local ffmpeg)
from pathlib import Path

import pytest


def ffmpeg_binary() -> str | None:
    """ffmpeg for the tests: $FRAMEKIT_TEST_FFMPEG (a path), else `ffmpeg` on PATH."""
    explicit = os.environ.get("FRAMEKIT_TEST_FFMPEG")
    if explicit and Path(explicit).is_file():
        return explicit
    return shutil.which("ffmpeg")


# CI sets FRAMEKIT_REQUIRE_FFMPEG=1, so a missing ffmpeg fails these tests there instead of
# skipping them silently.
needs_ffmpeg = pytest.mark.skipif(
    ffmpeg_binary() is None and os.environ.get("FRAMEKIT_REQUIRE_FFMPEG") != "1",
    reason="ffmpeg isn't installed",
)

SEGMENT_SOURCES = (
    "testsrc2=size={w}x{h}:rate={fps}",
    "smptebars=size={w}x{h}:rate={fps}",
    "color=c=0x3050a0:size={w}x{h}:rate={fps}",
    "rgbtestsrc=size={w}x{h}:rate={fps}",
)


def make_video(
    path: Path,
    *,
    cuts: tuple[float, ...] = (8.0, 16.0),
    duration: float = 24.0,
    beeps: tuple[float, ...] = (2.0, 5.0, 11.0, 19.0),
    size: tuple[int, int] = (640, 360),
    fps: int = 24,
    audio: bool = True,
) -> Path:
    """A video of visually distinct test patterns switching at `cuts`, with short 880 Hz
    beeps at `beeps` over silence."""
    binary = ffmpeg_binary()
    assert binary is not None
    edges = [0.0, *cuts, duration]
    inputs: list[str] = []
    for k, (a, b) in enumerate(zip(edges, edges[1:], strict=False)):
        source = SEGMENT_SOURCES[k % len(SEGMENT_SOURCES)].format(w=size[0], h=size[1], fps=fps)
        inputs += ["-f", "lavfi", "-i", f"{source}:duration={b - a}"]
    count = len(edges) - 1
    graph = "".join(f"[{k}:v]" for k in range(count)) + f"concat=n={count}:v=1:a=0[v]"
    args = [binary, "-hide_banner", "-nostdin", "-y", *inputs]
    maps = ["-map", "[v]"]
    if audio:
        gate = "+".join(f"between(t,{t},{t + 0.08})" for t in beeps) or "0"
        expression = f"0.8*sin(2*PI*880*t)*({gate})"
        args += ["-f", "lavfi", "-i", f"aevalsrc='{expression}':s=48000:d={duration}"]
        maps += ["-map", f"{count}:a", "-c:a", "aac"]
    args += [
        "-filter_complex",
        graph,
        *maps,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-t",
        str(duration),
        str(path),
    ]
    subprocess.run(args, check=True, capture_output=True)  # noqa: S603  (fixed arguments)
    return path
