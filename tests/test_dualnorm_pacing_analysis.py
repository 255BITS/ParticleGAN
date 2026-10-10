"""Reporting controls only: no GAN trajectory or optimizer update is executed."""
import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1] / "reports/forge/dualnorm-pacing-v2"
spec = importlib.util.spec_from_file_location("pacing_analysis_tests", HERE / "analyze.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
from experiments.forge.optimizer_diagnostics import OptimizerDiagnostics


def test_archive_fallback_uses_actual_campaign_and_durable_members(tmp_path):
    archive = tmp_path / "originals.tar.gz"
    contents = {"campaign/" + a.CAMPAIGN + "/id/trace.jsonl": b'{"step":1}\n',
                "campaign/" + a.CAMPAIGN + "/id/clockfree-proof/initial.pt": b'original-proof-bytes',
                "durable/id/evidence.json": b'{"result_hash":"bound"}\n'}
    with tarfile.open(archive, "w:gz") as writer:
        for name, payload in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            writer.addfile(info, BytesIO(payload))
    reader = a.Artifacts(tmp_path / "missing-root", tmp_path / "missing-queue", archive)
    try:
        assert reader.read("id", "trace.jsonl") == contents["campaign/" + a.CAMPAIGN + "/id/trace.jsonl"]
        assert reader.json("id", "evidence.json", True) == {"result_hash":"bound"}
        assert reader.read("id", "clockfree-proof/initial.pt") == b'original-proof-bytes'
        assert all(reader.refs[k]["sha256"] == hashlib.sha256(v).hexdigest() for k,v in contents.items())
    finally:
        reader.close()


def test_archive_reader_rejects_traversal(tmp_path):
    reader = a.Artifacts(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="within its attempt"):
        reader.read("id", "../../outside")


def test_word_previous_generator_phase_occupies_next_discriminator_cache_without_training(tmp_path):
    critic = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(critic.parameters(), lr=.01)
    optimizer.record = SimpleNamespace(observed_steps=41)
    observer = OptimizerDiagnostics({"D":optimizer}, critic, tmp_path, [42])
    # The preceding noncheckpoint D post-hook clears the cache. WordFixture
    # keeps D.train during the following G phase and sends fake then real.
    observer._after("D")(optimizer, (), {})
    critic.train()
    critic(torch.tensor([[100.]]))  # preceding G fake
    critic(torch.tensor([[200.]]))  # preceding G real
    critic(torch.tensor([[1.]]))    # current D real
    critic(torch.tensor([[2.]]))    # current D fake
    assert [float(x[0,0]) for x in observer.inputs] == [100., 200.]
    assert not a.input_gradient_scope(a.REQUIRED[5])["usable"]
    for handle in observer.handles:
        handle.remove()
    assert not (tmp_path / "optimizer-diagnostics.jsonl").exists()


def test_scalar_generator_eval_phase_leaves_current_real_fake_cache_correct(tmp_path):
    critic = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(critic.parameters(), lr=.01)
    optimizer.record = SimpleNamespace(observed_steps=41)
    observer = OptimizerDiagnostics({"D":optimizer}, critic, tmp_path, [42])
    observer._after("D")(optimizer, (), {})
    critic.eval()
    critic(torch.tensor([[100.]]))
    critic(torch.tensor([[200.]]))
    critic.train()
    critic(torch.tensor([[1.]]))
    critic(torch.tensor([[2.]]))
    assert [float(x[0,0]) for x in observer.inputs] == [1., 2.]
    assert a.input_gradient_scope(a.REQUIRED[0])["usable"]
    assert a.input_gradient_scope(a.REQUIRED[4])["usable"]
    for handle in observer.handles:
        handle.remove()


def test_word_gradient_fields_excluded_while_actual_updates_retained():
    row = {"step":1, "optimizer":"D", "players":{}, "layers":[],
           "critic_input_gradient_mean_real":123., "critic_input_gradient_mean_fake":456.}
    rows = a.clean_trace([row, {"step":1,"optimizer":"G","players":{},"layers":[]}], a.REQUIRED[5], [1])
    assert rows[0] == {k:v for k,v in row.items() if not k.startswith("critic_input_gradient_")}


def test_transient_and_short_terminal_passes_never_qualify():
    curve = [{"step":i,"v":0. if i in (15, 20) else 1.} for i in range(1,25)]
    verdict = a.test_verdict({"steps":24,"thresholds":[["v",">=",1.]]}, {"observations":curve,"live":curve[-1]})
    assert verdict["status"] == "FAIL"
    assert verdict["convergence"]["passing_suffix"] == 4
    assert verdict["metrics"][0]["status"] == "PASS"


def test_wrong_reduction_and_moved_unsampled_rows_are_rejected():
    rows = [{"optimizer":"D","step":1,"players":{"D":{"relative_update_sum":3.}},
             "layers":[{"player":"D","relative_update":1.}]}]
    with pytest.raises(ValueError, match="relative-speed"):
        a.clean_trace(rows, a.REQUIRED[0], [1])
    rows = [{"optimizer":"G","step":1,"players":{},"layers":[],"prior":{
             "row_support_kind":"sampled_indices","max_outside_support_row_displacement":.001}}]
    with pytest.raises(ValueError, match="unsampled"):
        a.clean_trace(rows, a.REQUIRED[0], [1])


def test_clock_parity_cannot_hide_unexplained_source_dependencies():
    comparisons = [{"reference_sha256":"same", "changed_sha256":"same"} for _ in range(4)]
    with pytest.raises(ValueError, match="unexplained clock"):
        a.task_summary({"tasks":{a.CLOCK:{}}}, {"evidence":{}}, {"task":a.CLOCK},
                       (comparisons,{"unexplained_clock_dependencies":["unapproved clock law"]}))
