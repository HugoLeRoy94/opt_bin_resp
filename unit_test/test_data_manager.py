import csv
from pathlib import Path

import pytest

import manage_data


def test_curation_round_trip_and_sorted_output(tmp_path, monkeypatch):
    catalog = tmp_path / "curation.csv"
    monkeypatch.setattr(manage_data, "CURATION_FILE", catalog)
    decisions = {
        "z_goal/sweep_20260101_000002": ("delete", ""),
        "a_goal/sweep_20260101_000001": ("keep", "reference"),
    }

    manage_data.save_curation(decisions)

    assert manage_data.load_curation() == decisions
    with catalog.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["path"] for row in rows] == sorted(decisions)


def test_kept_sweep_requires_label(tmp_path, monkeypatch):
    catalog = tmp_path / "curation.csv"
    catalog.write_text(
        "path,state,label\n"
        "goal/sweep_20260101_000001,keep,\n"
    )
    monkeypatch.setattr(manage_data, "CURATION_FILE", catalog)

    with pytest.raises(ValueError, match="needs a label"):
        manage_data.load_curation()


@pytest.mark.parametrize(
    "path",
    ["goal", "../goal/sweep_20260101_000001", "goal/not_timestamped", "/"],
)
def test_sweep_paths_are_strict(path):
    with pytest.raises(ValueError):
        manage_data._validate_sweep_path(path)


def test_execution_summary_supports_legacy_and_marker(tmp_path):
    sweep = tmp_path / "sweep_20260101_000001"
    complete = sweep / "run_1"
    partial = sweep / "run_2"
    complete.mkdir(parents=True)
    partial.mkdir()
    (complete / "config.json").write_text("{}")
    (complete / "test_results.json").write_text("{}")
    (partial / "config.json").write_text("{}")

    assert manage_data._execution_summary(sweep) == ("incomplete (legacy)", 1, 2)
    (sweep / ".state").write_text("failed\n")
    assert manage_data._execution_summary(sweep) == ("failed", 1, 2)
