"""Structure (question 4, docs/settings.md §5.3): what one data point is. Scenes (L2),
fixed time windows, and the beats inside a scene."""

from typing import ClassVar, Literal

from pydantic import Field

from framekit.extraction.schemas.common import Model
from framekit.extraction.schemas.plan.component import Component

# --- structure --------------------------------------------------------------------------


class SceneLength(Model):
    min: float = Field(default=30, gt=0)
    target: float = Field(default=45, gt=0)
    max: float = Field(default=60, gt=0)


class FixedWindows(Model):
    window_s: float | None = Field(default=None, gt=0)  # derived: the shortest moment length
    stride_s: float = Field(default=1, gt=0)


class Segmentation(Model):
    """L2. `method: model` asks the L2 model for a synopsis and scene boundaries (snapped to
    shot cuts); `shot_groups` groups consecutive shots into scenes of `scene_length_s`
    without a model (and without a synopsis)."""

    mode: Literal["scenes", "fixed_windows", "both"] = "scenes"
    method: Literal["model", "shot_groups"] = "model"
    scene_length_s: SceneLength = SceneLength()
    snap_to: Literal["shot_boundaries", "none"] = "shot_boundaries"
    snap_tolerance_s: float = Field(default=2.0, ge=0, le=30)
    fixed_windows: FixedWindows = FixedWindows()


class Beats(Model):
    min_s: float | None = Field(default=None, gt=0)  # derived when unset
    max_s: float | None = Field(default=None, gt=0)


# --- the component ----------------------------------------------------------------------


class Structure(Component):
    """What one data point is: `segmentation` (scenes, fixed windows) and `beats` (their
    length inside a scene; L3a writes them)."""

    KEY: ClassVar[str] = "structure"

    version: Literal[1] = 1
    segmentation: Segmentation = Segmentation()
    beats: Beats = Beats()

    def problems(self) -> list[str]:
        problems = []
        length = self.segmentation.scene_length_s
        if not length.min <= length.target <= length.max:
            problems.append("segmentation.scene_length_s: expected min <= target <= max")
        if self.segmentation.mode == "fixed_windows":
            problems.append(
                "segmentation.mode: fixed_windows alone leaves no scenes to perceive "
                "(perception and lens passes run on scenes): use both"
            )
        beats = self.beats
        if beats.min_s is not None and beats.max_s is not None and beats.min_s > beats.max_s:
            problems.append("beats: min_s is more than max_s")
        return problems
