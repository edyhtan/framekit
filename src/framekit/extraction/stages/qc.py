"""L4 · Quality control (DESIGN.md §3): results as `qc` rows, part of the provenance.

Library checks (times inside the scene and ordered, names from the profile, modules
complete), lens checks (schema, vocabulary, the lens's validators) and self-consistency
agreement. A failing row is flagged (or down-weighted), never silently dropped; passes
are recorded too, so a reader can see what was checked.
"""

from typing import Any

from framekit.extraction.schemas.lens import LensSpec
from framekit.extraction.stages.context import SceneRow

RECORD_CHECKS = (
    "timestamps_in_scene",
    "timestamps_ordered",
    "entities_in_profile",
    "modules_complete",
)


def qc_row(
    check: str,
    *,
    target_type: str,
    target_id: str,
    unit_id: str,
    status: str,
    detail: str = "",
    scene_id: str | None = None,
    lens: str | None = None,
    run_index: int | None = None,
) -> dict[str, Any]:
    return {
        "check": check,
        "target_type": target_type,
        "target_id": target_id,
        "unit_id": unit_id,
        "scene_id": scene_id,
        "lens": lens,
        "run_index": run_index,
        "status": status,
        "detail": detail,
    }


def _detail(problems: list[str], limit: int = 10) -> str:
    shown = problems[:limit]
    more = f" (+{len(problems) - limit} more)" if len(problems) > limit else ""
    return "; ".join(shown) + more


def segmentation_qc(
    unit_id: str, output: dict[str, Any], enabled: tuple[str, ...], then: str
) -> list[dict[str, Any]]:
    if "schema" not in enabled or output.get("method") != "model":
        return []
    flagged = output.get("status") == "flagged"
    return [
        qc_row(
            "schema",
            target_type="unit",
            target_id=unit_id,
            unit_id=unit_id,
            status=then if flagged else "pass",
            detail=_detail(output.get("problems") or []) if flagged else "L2",
        )
    ]


def perception_qc(
    scene: SceneRow, output: dict[str, Any], enabled: tuple[str, ...], then: str
) -> list[dict[str, Any]]:
    common = {
        "target_type": "scene",
        "target_id": scene.scene_id,
        "unit_id": scene.unit_id,
        "scene_id": scene.scene_id,
    }
    rows = []
    if "schema" in enabled:
        flagged = output.get("record") is None
        rows.append(
            qc_row(
                "schema",
                status=then if flagged else "pass",
                detail=_detail(output.get("problems") or []) if flagged else "L3a",
                **common,
            )
        )
    for check, problems in (output.get("checks") or {}).items():
        if check in enabled:
            rows.append(
                qc_row(
                    check, status="flag" if problems else "pass", detail=_detail(problems), **common
                )
            )
    return rows


def lens_qc(
    scene: SceneRow,
    lens: LensSpec,
    run_index: int,
    output: dict[str, Any],
    enabled: tuple[str, ...],
    then: str,
) -> list[dict[str, Any]]:
    common = {
        "target_type": "scene",
        "target_id": scene.scene_id,
        "unit_id": scene.unit_id,
        "scene_id": scene.scene_id,
        "lens": lens.name,
        "run_index": run_index,
    }
    rows = []
    annotation = output.get("annotation")
    problems = output.get("problems") or []
    if annotation is None:
        vocabulary = [p for p in problems if "vocabulary" in p]
        if "schema" in enabled:
            rows.append(qc_row("schema", status=then, detail=_detail(problems), **common))
        if "vocabulary" in enabled:
            rows.append(
                qc_row(
                    "vocabulary",
                    status=then if vocabulary else "pass",
                    detail=_detail(vocabulary),
                    **common,
                )
            )
        return rows
    if "schema" in enabled:
        rows.append(qc_row("schema", status="pass", **common))
    if "vocabulary" in enabled:
        rows.append(qc_row("vocabulary", status="pass", **common))
    if "validators" in enabled and lens.validators:
        broken = []
        if "scene" in annotation:
            broken += lens.check_rules(annotation["scene"], "scene")
        for beat in annotation.get("beats") or ():
            broken += lens.check_rules(beat, f"beats[{beat.get('index')}]")
        rows.append(
            qc_row(
                "validators", status="flag" if broken else "pass", detail=_detail(broken), **common
            )
        )
    return rows


def agreement_qc(
    scene: SceneRow, lens: LensSpec, score: float, threshold: float, on_disagreement: str
) -> dict[str, Any]:
    return qc_row(
        "agreement",
        target_type="scene",
        target_id=scene.scene_id,
        unit_id=scene.unit_id,
        scene_id=scene.scene_id,
        lens=lens.name,
        status="pass" if score >= threshold else on_disagreement,
        detail=f"agreement {score:.2f} (min {threshold:.2f})",
    )


__all__ = ["agreement_qc", "lens_qc", "perception_qc", "qc_row", "segmentation_qc"]
