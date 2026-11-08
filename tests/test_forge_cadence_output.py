"""Generated cadence exports must never replace source or qualification files."""
from copy import deepcopy
import json

import pytest

from experiments.forge.cadence_study import write_report


@pytest.fixture
def report():
    return dict(artifact_kind="forge_cadence_comparison_v1", schema_version=1,
                study_id="lazy-study", hypothesis="Preserve the eager gates.",
                stage="planned", declared_worst_case_seconds=40,
                interpretation="Fixed protocol; no training launched.", lanes={})


def snapshot(root):
    return {str(path.relative_to(root)): ("link", str(path.readlink())) if path.is_symlink()
            else ("file", path.read_bytes()) if path.is_file() else ("directory", None)
            for path in root.rglob("*")}


def test_new_explicit_export_and_owned_pair_refresh(report, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = write_report(report, "exports/readout.md")
    output = tmp_path / "exports/readout.md"
    companion = output.with_suffix(".json")
    assert result == dict(markdown=str(output), json=str(companion))
    assert output.read_text().startswith("# lazy-study\n")
    assert json.loads(companion.read_text()) == report
    changed = {**report, "hypothesis": "Updated generated readout.",
               "cost_note": "Shared campaign costs are not additive across lanes."}
    write_report(changed, output)
    assert changed["hypothesis"] in output.read_text()
    assert changed["cost_note"] in output.read_text()
    assert json.loads(companion.read_text()) == changed


@pytest.mark.parametrize("destination,existing", [
    ("reports/forge/technique-inventory.md", "both"),
    ("configs/forge/cadence-studies/lazy-study.md", "json"),
    ("reports/forge/cadence-studies/lazy-study-registration.md", "json"),
    ("EXPERIMENTATION.md", "md"),
])
def test_existing_source_companion_and_leaderboard_are_preserved(report, tmp_path, destination, existing):
    output = tmp_path / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    if existing in {"md", "both"}:
        output.write_text("# Original document\nEvidence remains unchanged.\n")
    if existing in {"json", "both"}:
        output.with_suffix(".json").write_text('{"source": "immutable"}\n')
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_report(report, output)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("mutation", ["other_study", "wrong_kind", "wrong_header", "invalid_json",
                                       "missing_md", "missing_json"])
def test_only_a_complete_matching_generated_pair_can_be_refreshed(report, tmp_path, mutation):
    output = tmp_path / "comparison.md"
    write_report(report, output)
    companion = output.with_suffix(".json")
    if mutation in {"other_study", "wrong_kind"}:
        existing = deepcopy(report)
        existing["study_id" if mutation == "other_study" else "artifact_kind"] = "foreign"
        companion.write_text(json.dumps(existing))
    elif mutation == "wrong_header":
        output.write_text("# Foreign study\n")
    elif mutation == "invalid_json":
        companion.write_text("not json\n")
    elif mutation == "missing_md":
        output.unlink()
    else:
        companion.unlink()
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_report(report, output)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("linked", ["markdown", "companion", "parent"])
def test_symlinks_cannot_hide_foreign_files(report, tmp_path, linked):
    source = tmp_path / "source"
    source.mkdir()
    (source / "report.md").write_text("# Preserved source\n")
    (source / "report.json").write_text('{"source": true}\n')
    export = tmp_path / "export"
    if linked == "parent":
        export.symlink_to(source, target_is_directory=True)
    else:
        export.mkdir()
        (export / ("report.md" if linked == "markdown" else "report.json")).symlink_to(
            source / ("report.md" if linked == "markdown" else "report.json"))
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_report(report, export / "report.md")
    assert snapshot(tmp_path) == before


def test_owned_symlink_pair_refreshes_resolved_targets(report, tmp_path):
    output = tmp_path / "actual/report.md"
    write_report(report, output)
    links = tmp_path / "links"
    links.mkdir()
    (links / "readout.md").symlink_to(output)
    (links / "readout.json").symlink_to(output.with_suffix(".json"))
    changed = {**report, "stage": "reported"}
    result = write_report(changed, links / "readout.md")
    assert result == dict(markdown=str(output), json=str(output.with_suffix(".json")))
    assert (links / "readout.md").is_symlink()
    assert json.loads(output.with_suffix(".json").read_text()) == changed


@pytest.mark.parametrize("mutation", ["wrong_kind", "nonfinite", "wrong_suffix", "render_error"])
def test_invalid_exports_create_no_files_or_directories(report, tmp_path, mutation):
    output = tmp_path / "new/deep/report.md"
    if mutation == "wrong_kind":
        report["artifact_kind"] = "source"
    elif mutation == "nonfinite":
        report["extra"] = float("nan")
    elif mutation == "wrong_suffix":
        output = output.with_suffix(".py")
    else:
        report["lanes"] = {"ordinary": {}}
    before = snapshot(tmp_path)
    with pytest.raises((ValueError, KeyError)):
        write_report(report, output)
    assert snapshot(tmp_path) == before
