"""L0 · Normalize (DESIGN.md §3): the analysis copy of a unit, in the work folder.

The source is only read. The copy is a cacheable processing artifact keyed by the source's
content fingerprint and the normalize settings; for a byo_media corpus it counts as
footage, so it stays in the work folder (which holds a `.gitignore` of `*`).
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from framekit.extraction.media.ffmpeg import NORMALIZE_VERSION, MediaInfo
from framekit.extraction.resolve import ScopedUnit
from framekit.extraction.runner import ensure_private_dir
from framekit.extraction.schemas.common import plain
from framekit.extraction.stages.context import StageContext, UnitMedia, quick_fingerprint


async def probe_source(ctx: StageContext, unit: ScopedUnit) -> tuple[MediaInfo, str]:
    """The source's facts and content fingerprint (cheap: no decoding)."""
    path = Path(unit.unit.path)
    async with ctx.ex.media_slots:
        info = await ctx.ffmpeg.probe(path)
    return info, quick_fingerprint(path)


async def normalize(
    ctx: StageContext, unit: ScopedUnit, source: MediaInfo, fingerprint: str
) -> tuple[UnitMedia, dict[str, Any]]:
    settings = ctx.run.plan.normalize
    media_dir = ctx.media_dir
    inputs = {"version": NORMALIZE_VERSION, "source": fingerprint, "settings": plain(settings)}

    async def compute(input_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
        ensure_private_dir(media_dir)
        target = media_dir / f"{unit.unit_id}-{input_hash[:16]}.mp4"
        async with ctx.ex.media_slots:
            info = await ctx.ffmpeg.normalize(unit.unit.path, target, settings, source)
        output = {
            "copy": target.name,
            "info": asdict(info),
            "copy_fingerprint": quick_fingerprint(target),
        }
        return output, {"unit_id": unit.unit_id, "target_id": unit.unit_id}

    output, record = await ctx.ex.task(
        "L0", inputs, compute, valid=lambda o: (media_dir / o["copy"]).is_file()
    )
    media = UnitMedia(
        unit=unit,
        source=source,
        source_fingerprint=fingerprint,
        copy_path=media_dir / output["copy"],
        copy=MediaInfo(**output["info"]),
        copy_fingerprint=output["copy_fingerprint"],
    )
    return media, record


__all__ = ["normalize", "probe_source"]
