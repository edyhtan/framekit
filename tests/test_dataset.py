"""The dataset format (docs/settings.md §7): fixed table schemas, the writer in either layout,
the reader, checksums and the schema version."""

from pathlib import Path

import pyarrow as pa
import pytest

from framekit.extraction.dataset import (
    SCHEMAS,
    Dataset,
    prepare_output,
    to_table,
    write_manifest,
    write_tables,
)
from framekit.extraction.schemas.manifest import DatasetManifest


def test_tables_have_fixed_schemas():
    with pytest.raises(ValueError, match="unknown columns"):
        to_table("scenes", [{"scene_id": "s", "colour": "red"}])
    table = to_table("annotations", [{"field": "f", "value": {"a": 1}}])
    assert table.column("value").to_pylist() == ['{"a":1}']  # free-form values are JSON


def manifest(**extra):
    fields = {
        "schema_version": 1,
        "dataset_id": "ds_x",
        "created_at": "now",
        "framekit_version": "0",
        "complete": True,
        "search_space": {},
        "corpus": {},
        "plan": {},
    }
    return DatasetManifest(**(fields | extra))


@pytest.mark.parametrize("partition_by", ["unit_id", "none"])
def test_tables_round_trip_in_either_layout(tmp_path, partition_by):
    rows = {
        "scenes": [
            {"scene_id": f"u{k}_s000", "unit_id": f"u{k}", "index": 0, "start": 0.0, "end": 5.0}
            for k in (1, 2)
        ]
    }
    written = write_tables(
        tmp_path, rows, include=["scenes", "qc"], partition_by=partition_by, compression="zstd"
    )
    write_manifest(tmp_path, manifest(tables=written))
    dataset = Dataset.open(tmp_path, verify=True)
    assert dataset.scenes.num_rows == 2 and dataset.qc.num_rows == 0
    assert dataset.qc.schema == SCHEMAS["qc"]
    files = [f.path for f in written["scenes"].files]
    expected = (
        ["tables/scenes/unit-u1.parquet", "tables/scenes/unit-u2.parquet"]
        if partition_by == "unit_id"
        else ["tables/scenes.parquet"]
    )
    assert files == expected


def test_a_changed_file_or_an_unknown_schema_version_is_refused(tmp_path):
    written = write_tables(
        tmp_path, {"scenes": []}, include=["scenes"], partition_by="none", compression="none"
    )
    write_manifest(tmp_path, manifest(tables=written))
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"x": [1]}), tmp_path / "tables" / "scenes.parquet")
    with pytest.raises(ValueError, match="changed since it was written"):
        Dataset.open(tmp_path, verify=True)
    write_manifest(tmp_path, manifest(schema_version=9))
    with pytest.raises(ValueError, match="schema version 9"):
        Dataset.open(tmp_path)
    with pytest.raises(FileNotFoundError, match="not a finished dataset"):
        Dataset.open(Path(tmp_path) / "elsewhere")


def test_an_output_folder_holding_another_dataset_is_refused(tmp_path):
    identity = {"search_space": "a", "corpus": "b", "plan": "c"}
    prepare_output(tmp_path / "out", identity)
    assert (tmp_path / "out" / ".gitignore").read_text() == "*\n"
    write_manifest(
        tmp_path / "out",
        manifest(search_space={"digest": "a"}, corpus={"digest": "b"}, plan={"digest": "c"}),
    )
    prepare_output(tmp_path / "out", identity)  # the same inputs: a rerun may rewrite it
    with pytest.raises(FileExistsError, match="holds another dataset"):
        prepare_output(tmp_path / "out", {**identity, "plan": "d"})
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError, match="isn't empty"):
        prepare_output(tmp_path / "other", identity)
