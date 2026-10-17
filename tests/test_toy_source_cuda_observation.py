"""Software purity and unchanged-budget controls; no quality training."""
from __future__ import annotations

import importlib.util
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.toy_audit import source_cuda_observation as observer


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Root runs this control with one visible GPU"))])
def test_rng_modes_gradients_and_nested_named_streams_are_preserved(device):
    module = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.5)).to(device)
    module[0].eval()  # mixed flags must stay mixed
    optimizer = torch.optim.Adam(module.parameters(), lr=.01)
    cpu_stream = torch.Generator().manual_seed(17)
    device_stream = torch.Generator(device=device).manual_seed(23)
    local = dict(module=module, optimizers=[optimizer], nested={"rngs": [cpu_stream, device_stream]})
    original = observer.digest(observer.owners(local))

    def read():
        random.random(); np.random.random(); torch.rand(5)
        if device != "cpu":
            torch.rand(5, device=device)  # global CUDA stream must be restored
        private = torch.Generator(device=device).manual_seed(51)
        return module(torch.randn(9, 3, device=device, generator=private))

    result, before = observer.pure_read(local, read)
    assert result.shape == (9, 3) and before == original
    assert observer.digest(observer.owners(local)) == original
    assert module.training and not module[0].training and module[1].training
    if device != "cpu":
        assert len(observer.owners(local)["global_rng"]["cuda"]) == torch.cuda.device_count()


def test_mutating_model_or_owned_stream_is_rejected():
    module = torch.nn.Linear(2, 2)
    stream = torch.Generator().manual_seed(19)
    with pytest.raises(AssertionError, match="observer changed"):
        observer.pure_read(dict(module=module), lambda: module.weight.add_(1))
    with pytest.raises(AssertionError, match="observer changed"):
        observer.pure_read(dict(nested={"named_stream": stream}), lambda: torch.randn(3, generator=stream))


def test_digest_supports_scalar_and_bfloat16_states():
    assert observer.digest(torch.tensor(2.)) == observer.digest(torch.tensor(2.))
    assert observer.digest(torch.tensor([1., 2.], dtype=torch.bfloat16)) != observer.digest(torch.tensor([2., 1.], dtype=torch.bfloat16))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Root runs actual CUDA owner-prefix control"))])
def test_prefix_observation_preserves_updates_and_full_schedule(tmp_path, monkeypatch, device):
    source = tmp_path / "held_source.py"
    source.write_text('''import torch
def train(cfg):
    torch.manual_seed(241)
    device = cfg['device']
    if device != 'cpu':
        torch.cuda.manual_seed_all(241)
    model = torch.nn.Linear(2, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    rngs = {'data': torch.Generator(device=device).manual_seed(244)}
    for step in range(1, cfg['steps'] + 1):
        x = torch.randn(5, 2, device=device, generator=rngs['data'])
        target = torch.randn(5, 2, device=device)
        optimizer.param_groups[0]['lr'] = .01 * (1 - (step - 1) / cfg['steps'])
        loss = (model(x) - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {'steps': cfg['steps']}
''')
    spec = importlib.util.spec_from_file_location("_prefix_fixture", source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cfg = dict(steps=17, device=device)
    observed_steps = []

    def cheap_full_law_read(capture, step, local):
        observer.pure_read(local, lambda: torch.randn(6, device=device))
        observed_steps.append(step)
        capture.rows.append(dict(step=step))

    monkeypatch.setattr(observer.Capture, "observe", cheap_full_law_read)
    entry_rng = observer.global_rng()
    baseline = observer.execute(module, cfg, "denoising", tmp_path / "baseline", observe=False, prefix=True)
    observer.restore_global_rng(entry_rng)
    observed = observer.execute(module, cfg, "denoising", tmp_path / "observed", observe=True, prefix=True)
    assert baseline["status"] == observed["status"] == "PREFIX_COMPLETE"
    assert baseline["boundaries"] == observed["boundaries"]
    assert set(baseline["boundaries"]) == {0, 1, 2, 3}
    assert observed_steps == [1]
    assert baseline["original_budget"] == observed["original_budget"] == cfg["steps"] == 17
    assert baseline["completed_updates"] == observed["completed_updates"] == 3
    assert not baseline["complete_original_endpoint"] and not observed["complete_original_endpoint"]


def test_registered_source_configs_and_budgets_remain_exact():
    assert observer.ENTRIES == {
        "source-family-02": ("denoising", "configs/denoising/diagnostics/ddgan_class_free_28k_s24002.yaml", 28000),
        "source-family-03": ("denoising", "configs/denoising/default.toml", 7000),
        "source-family-04": ("trajectory", "configs/trajectory/default.yaml", 10000),
        "source-family-05": ("trajectory", "configs/trajectory/diversity/confirm_10k/mlp_continuous.yaml", 10000),
    }
    assert observer.SOURCE_SHA == "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
    assert observer.WALL_CAP == 120
    assert observer.scoring.COUNT == 4096 and observer.scoring.EVAL_SEED == 99123


def test_good_prefix_or_finished_updates_without_source_endpoint_cannot_pass():
    from benchmarks.toy_audit.source_cuda_observation_report import assess
    rows = [dict(step=k, metrics=dict(live=dict(passed=True), ema=dict(passed=True))) for k in (0, 10, 20, 30, 40)]
    paid = dict(complete_original_endpoint=False, completed_updates=40)
    complete, added, suffix = assess("denoising", "INCOMPLETE", paid, rows, 100)
    assert not complete and added == dict(live="INCOMPLETE", ema="INCOMPLETE")
    assert suffix["live"]["passed"] and suffix["ema"]["passed"]
    # Even every prescribed update is insufficient when the source's final
    # diagnostic/render/save was interrupted by the cap.
    paid["completed_updates"] = rows[-1]["step"] = 100
    assert assess("denoising", "INCOMPLETE", paid, rows, 100)[1]["live"] == "INCOMPLETE"
    paid["complete_original_endpoint"] = True
    assert assess("denoising", "COMPLETE", paid, rows, 100)[1] == dict(live="PASS", ema="PASS")
    rows[-2]["metrics"]["live"]["passed"] = False
    assert assess("denoising", "COMPLETE", paid, rows, 100)[1] == dict(live="FAIL", ema="PASS")
    assert assess("trajectory", "COMPLETE", paid, rows, 100)[1].startswith("NO_FROZEN_GATE")


def test_actual_import_resolution_requires_both_held_path_and_bytes(tmp_path):
    source = tmp_path / "held"
    source.mkdir()
    held = source / "helper.py"
    held.write_text("# held source\n")
    expected = {"helper.py": observer.sha(held)}
    registry = {"lib.helper": SimpleNamespace(__file__=str(held))}
    assert observer.imported_source_bindings(source, expected, modules=registry)["lib.helper"]["sha256"] == expected["helper.py"]
    escaped = tmp_path / "helper.py"
    escaped.write_bytes(held.read_bytes())
    with pytest.raises(AssertionError, match="escaped held source"):
        observer.imported_source_bindings(source, expected, modules={"lib.helper": SimpleNamespace(__file__=str(escaped))})
    held.write_text("# modified source\n")
    with pytest.raises(AssertionError, match="imported bytes changed"):
        observer.imported_source_bindings(source, expected, modules=registry)
