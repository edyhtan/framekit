"""The `framekit` command: a dry run with sources from a module, the cache commands,
configuration problems reported as messages, and the option parsers."""

from pathlib import Path

from extraction_helpers import write_inputs, write_yaml
from media_helpers import ffmpeg_binary, make_video, needs_ffmpeg

from framekit.cli import main, parse_cache_modes, parse_rate
from framekit.models.cache import ClassPolicy


@needs_ffmpeg
def test_the_cli_runs_a_dry_run_and_reads_the_cache(tmp_path, capsys, monkeypatch):
    video = make_video(tmp_path / "u1.mp4")
    files = write_inputs(tmp_path, video)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    code = main(
        [
            "extract",
            "--space",
            str(files["space"]),
            "--corpus",
            str(files["corpus"]),
            "--plan",
            str(files["plan"]),
            "--out",
            str(tmp_path / "cli"),
            "--dry-run",
            "--ffmpeg",
            ffmpeg_binary(),
            "--sources",
            "model_helpers:sources",
        ]
    )
    assert code == 0 and "Estimate (upper bound)" in capsys.readouterr().out
    assert main(["cache", "stats", "--cache-dir", str(tmp_path / "empty")]) == 0
    assert "the cache is empty" in capsys.readouterr().out


def test_cli_config_errors_are_reported_not_raised(tmp_path, capsys):
    bad = write_yaml(tmp_path / "space.yaml", {"search_space": {"lenses": []}})
    code = main(
        ["extract", "--space", str(bad), "--corpus", str(bad), "--out", str(tmp_path / "x")]
    )
    assert code == 1 and "problem" in capsys.readouterr().err


def test_cache_policies_per_class_come_from_the_cli():
    assert parse_cache_modes(["record", "L3b=replay"]) == ("record", {"L3b": "replay"})
    assert parse_rate("hive=120/m") == ("hive", 2.0)
    assert ClassPolicy(mode="replay").mode == "replay"
