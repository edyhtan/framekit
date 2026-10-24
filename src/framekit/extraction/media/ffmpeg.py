"""The ffmpeg binary, behind small async functions: probe, normalize, decode, encode.

framekit needs only `ffmpeg` (not ffprobe). It's run as a subprocess with an argument list,
never through a shell, so file names can't inject commands. Which binary is execution, not
plan: pass a path (`LocalRunner(ffmpeg=...)`) or have `ffmpeg` on PATH. Its version is
recorded in each dataset's manifest.

Times are seconds on the unit timeline; the analysis copy starts at 0 like its source.
"""

import asyncio
import math
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from framekit.extraction.schemas.plan import NormalizeSettings

# Keyframes every 2 s in the analysis copy, so reading a frame decodes at most 2 s.
KEYFRAME_INTERVAL_S = 2.0
NORMALIZE_VERSION = "normalize@1"  # changes when the commands below change


class MediaError(RuntimeError):
    """ffmpeg failed or the file isn't usable media. `retryable` is False: the same command
    on the same file fails the same way."""

    retryable = False


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    width: int | None  # as displayed (rotation applied)
    height: int | None
    fps: float | None
    has_audio: bool
    sample_rate_hz: int | None = None
    channels: int | None = None
    video_codec: str | None = None

    @property
    def long_side_px(self) -> int:
        return max(self.width or 0, self.height or 0)


_DURATION = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_STREAM = re.compile(r"^\s*Stream #\d+:\d+.*?: (Video|Audio): (\w+)(.*)$")
_SIZE = re.compile(r",\s*(\d{2,5})x(\d{2,5})[\s,\[]")
_FPS = re.compile(r",\s*(\d+(?:\.\d+)?)\s*fps\b")
_TBR = re.compile(r",\s*(\d+(?:\.\d+)?)(k?)\s*tbr\b")
_RATE = re.compile(r",\s*(\d+)\s*Hz\b")
_ROTATION = re.compile(r"rotation of (-?\d+(?:\.\d+)?) degrees")
_LAYOUTS = {"mono": 1, "stereo": 2}


def parse_probe(stderr: str) -> MediaInfo:
    """The facts `ffmpeg -i file` prints: duration, the first video stream (cover art
    skipped) and the first audio stream. Raises MediaError when there's no duration or no
    video stream."""
    duration = _DURATION.search(stderr)
    if duration is None:
        raise MediaError("not a media file ffmpeg can read (no duration)")
    hours, minutes, seconds = duration.groups()
    info: dict = {"duration_s": int(hours) * 3600 + int(minutes) * 60 + float(seconds)}
    rotation = 0.0
    found_video = found_audio = False
    for line in stderr.splitlines():
        turn = _ROTATION.search(line)
        if turn and found_video:
            rotation = float(turn.group(1))
        stream = _STREAM.match(line)
        if stream is None:
            continue
        kind, codec, rest = stream.groups()
        if kind == "Video" and not found_video and "attached pic" not in rest:
            size = _SIZE.search(rest)
            fps = _FPS.search(rest)
            tbr = _TBR.search(rest)
            rate = float(fps.group(1)) if fps else None
            if rate is None and tbr:
                rate = float(tbr.group(1)) * (1000 if tbr.group(2) else 1)
            info |= {
                "video_codec": codec,
                "width": int(size.group(1)) if size else None,
                "height": int(size.group(2)) if size else None,
                "fps": rate,
            }
            found_video = True
        elif kind == "Audio" and not found_audio:
            rate = _RATE.search(rest)
            channels = next((n for name, n in _LAYOUTS.items() if f", {name}" in rest), None)
            if channels is None:
                count = re.search(r",\s*(\d+) channels", rest)
                channels = int(count.group(1)) if count else None
            info |= {
                "has_audio": True,
                "sample_rate_hz": int(rate.group(1)) if rate else None,
                "channels": channels,
            }
            found_audio = True
    if not found_video:
        raise MediaError("no video stream")
    if round(abs(rotation)) % 180 == 90 and info.get("width") and info.get("height"):
        info["width"], info["height"] = info["height"], info["width"]  # displayed size
    info.setdefault("has_audio", False)
    return MediaInfo(**info)


def copy_size(info: MediaInfo, height_px: int) -> tuple[int, int]:
    """The analysis copy's frame size: `height_px` tall at most (never upscaled), even
    sides, the displayed aspect kept."""
    if not info.width or not info.height:
        raise MediaError("the source has no frame size")
    height = min(height_px, info.height)
    height -= height % 2
    width = 2 * math.floor(info.width * height / info.height / 2 + 0.5)
    return max(width, 2), max(height, 2)


def scaled_size(width: int, height: int, max_side_px: int) -> tuple[int, int]:
    """A frame scaled so its long side is at most `max_side_px` (never upscaled), aspect
    kept: `image.aspect: preserve` (never cropped or stretched)."""
    factor = min(1.0, max_side_px / max(width, height))
    return max(1, round(width * factor)), max(1, round(height * factor))


def frame_index(t: float, fps: float) -> int:
    """The frame of a constant-rate copy on screen at time t."""
    return max(0, math.floor(t * fps + 1e-6))


class FFmpeg:
    """One ffmpeg binary. `binary` is a name looked up on PATH, or a path."""

    def __init__(self, binary: str | Path = "ffmpeg") -> None:
        found = shutil.which(str(binary))
        if found is None and Path(binary).expanduser().is_file():
            found = str(Path(binary).expanduser())
        if found is None:
            raise MediaError(
                f"ffmpeg not found ({binary!r}): install ffmpeg, or pass its path "
                "(LocalRunner(ffmpeg=...), or --ffmpeg on the command line)"
            )
        self.path = found
        self._version: str | None = None

    async def run(
        self, args: Sequence[str], *, stdin: bytes | None = None, check: bool = True
    ) -> tuple[bytes, str]:
        """Run ffmpeg; returns (stdout bytes, stderr text). With `check`, a non-zero exit
        raises MediaError carrying the end of stderr."""
        flags = ["-hide_banner"] if stdin is not None else ["-hide_banner", "-nostdin"]
        process = await asyncio.create_subprocess_exec(
            self.path,
            *flags,
            *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await process.communicate(stdin)
        text = err.decode("utf-8", "replace")
        if check and process.returncode != 0:
            tail = " | ".join(line.strip() for line in text.strip().splitlines()[-3:])
            raise MediaError(f"ffmpeg failed (exit {process.returncode}): {tail}")
        return out, text

    async def version(self) -> str:
        if self._version is None:
            out, _ = await self.run(["-version"])
            self._version = out.decode("utf-8", "replace").splitlines()[0].strip()
        return self._version

    async def probe(self, path: str | Path) -> MediaInfo:
        if not Path(path).is_file():
            raise MediaError(f"{path}: no such file")
        _, text = await self.run(["-i", str(path)], check=False)  # exits 1: no output file
        try:
            return parse_probe(text)
        except MediaError as exc:
            raise MediaError(f"{path}: {exc}") from None

    async def normalize(
        self, source: str | Path, target: str | Path, settings: NormalizeSettings, info: MediaInfo
    ) -> MediaInfo:
        """The analysis copy (L0): constant fps, at most `height_px` tall with the displayed
        aspect (square pixels), H.264 8-bit 4:2:0, mono or stereo AAC, no metadata,
        subtitles or data streams. The source is only read."""
        width, height = copy_size(info, settings.height_px)
        filters = []
        if settings.deinterlace == "auto":
            filters.append("bwdif=mode=send_frame:deint=interlaced")  # flagged frames only
        filters += [
            f"fps={settings.fps:g}",
            f"scale={width}:{height}:flags=area",
            "setsar=1",
        ]
        if settings.color.normalize == "auto":
            filters.append("format=yuv420p")
        args = [
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            ",".join(filters),
            "-r",
            f"{settings.fps:g}",
            "-c:v",
            "libx264",
            "-preset",
            settings.encoding.preset,
            "-crf",
            str(settings.encoding.crf),
            "-g",
            str(max(1, round(KEYFRAME_INTERVAL_S * settings.fps))),
        ]
        if info.has_audio:
            args += [
                "-map",
                "0:a:0",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                str(settings.audio.sample_rate_hz),
                "-ac",
                str(settings.audio.channels),
            ]
        else:
            args += ["-an"]
        args += ["-map_metadata", "-1", "-sn", "-dn", "-movflags", "+faststart", str(target)]
        await self.run(args)
        return await self.probe(target)

    async def scene_scores(self, path: str | Path, width: int = 160) -> list[tuple[float, float]]:
        """ffmpeg's scene-change score (0–1) of every frame, as (time_s, score)."""
        _, text = await self.run(
            [
                "-nostats",
                "-i",
                str(path),
                "-an",
                "-sn",
                "-vf",
                f"scale={width}:-2,select='gte(scene,0)',metadata=print:key=lavfi.scene_score",
                "-f",
                "null",
                "-",
            ]
        )
        scores = []
        time = None
        for line in text.splitlines():
            if "pts_time:" in line:
                time = float(line.rsplit("pts_time:", 1)[1].split()[0])
            elif "lavfi.scene_score=" in line and time is not None:
                scores.append((time, float(line.rsplit("=", 1)[1])))
                time = None
        return scores

    async def audio_samples(self, path: str | Path, rate_hz: int = 16000) -> bytes:
        """The first audio stream as mono signed 16-bit PCM at `rate_hz` (b'' without one)."""
        out, _ = await self.run(
            [
                "-i",
                str(path),
                "-vn",
                "-sn",
                "-map",
                "0:a:0?",
                "-ac",
                "1",
                "-ar",
                str(rate_hz),
                "-f",
                "s16le",
                "-",
            ]
        )
        return out

    async def gray_frames(
        self, path: str | Path, fps: float, size: tuple[int, int]
    ) -> tuple[bytes, int]:
        """Frames at `fps` as raw 8-bit gray of `size` (width, height); returns the bytes
        and the frame count. Frame k is at k / fps."""
        width, height = size
        out, _ = await self.run(
            [
                "-i",
                str(path),
                "-an",
                "-sn",
                "-vf",
                f"fps={fps:g},scale={width}:{height}:flags=area,format=gray",
                "-f",
                "rawvideo",
                "-",
            ]
        )
        return out, len(out) // (width * height)

    async def frame_rgb(
        self, path: str | Path, index: int, fps: float, size: tuple[int, int]
    ) -> bytes:
        """Frame `index` of a constant-rate copy, as raw RGB of `size`. Seeks to a quarter
        frame before it, so the first frame decoded at or after that point is the one."""
        width, height = size
        seek = max(0.0, (index - 0.25) / fps)
        out, _ = await self.run(
            [
                "-ss",
                f"{seek:.6f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-vf",
                f"scale={width}:{height}:flags=area",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
        )
        if len(out) != width * height * 3:
            raise MediaError(f"{path}: no frame {index} (at {index / fps:.3f} s)")
        return out

    async def encode_image(
        self, rgb: bytes, size: tuple[int, int], fmt: str, quality: int | None
    ) -> bytes:
        """Raw RGB to an image file's bytes: jpeg (quality 1–100 → ffmpeg's MJPEG scale
        31–2), png, or webp (quality as given)."""
        width, height = size
        args = [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-i",
            "-",
            "-frames:v",
            "1",
        ]
        if fmt == "jpeg":
            args += [
                "-c:v",
                "mjpeg",
                "-q:v",
                str(jpeg_scale(quality or 85)),
                "-pix_fmt",
                "yuvj420p",
                "-f",
                "image2pipe",
                "-",
            ]
        elif fmt == "png":
            args += ["-c:v", "png", "-f", "image2pipe", "-"]
        elif fmt == "webp":
            args += ["-c:v", "libwebp", "-quality", str(quality or 85), "-f", "webp", "-"]
        else:
            raise MediaError(f"unknown image format {fmt!r}")
        out, _ = await self.run(args, stdin=rgb)
        return out


def jpeg_scale(quality: int) -> int:
    """JPEG quality 1–100 to ffmpeg's MJPEG quantizer scale (2 best, 31 worst), linearly."""
    return min(31, max(2, round(2 + (100 - quality) * 29 / 99)))


__all__ = [
    "KEYFRAME_INTERVAL_S",
    "NORMALIZE_VERSION",
    "FFmpeg",
    "MediaError",
    "MediaInfo",
    "copy_size",
    "frame_index",
    "jpeg_scale",
    "parse_probe",
    "scaled_size",
]
