"""Software controls, synthetic construction inputs, no training or fitting."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


_PATH = Path(__file__).with_name("bind_capacity.py")
_SPEC = importlib.util.spec_from_file_location("generator_step_capacity", _PATH)
binder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(binder)


def _rebind(record, role):
    path = Path(record["artifacts"][role]["path"])
    record["artifacts"][role].update(sha256=binder._hash(path), bytes=path.stat().st_size)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Fresh public source host; its initial random G is a negative witness.

    A fixed synthetic input/source oracle replaces only local historical
    artifacts and Git history. Real public construction, sampler, scorer,
    checkpoint loading, clocks, ownership and purity remain unmocked.
    """
    torch.set_num_threads(1)
    case = binder.selected_cases()["image-develop-img_intensity2-source-transpose12"]
    old_inputs = {}
    for family in binder.FAMILIES:
        with binder.api_run.isolated_evaluation():
            torch.manual_seed(binder.SEED)
            fixture = binder.contract.build(case, device="cpu", seed=binder.SEED, recipe_name=family,
                                             max_steps=case["default_steps"],
                                             recipe_overrides={"lr": .0053125, "prior_lr_mult": 1.5})
            state = fixture.state_dict()
        # Deliberately unrelated old averages: they must never be imported.
        for tensor in state["api_state"]["models"]["ema_G"].values():
            tensor.fill_(17)
        path = tmp_path / f"{family}-synthetic-input.pt"
        torch.save(state, path)
        original = dict(family=family, case_id=case["id"],
                        bindings={"case_sha256": binder.study.digest(case)},
                        artifacts={"state": {"path": str(path), "sha256": binder._hash(path)}})
        old_inputs[family] = dict(card={"path": str(tmp_path / "synthetic-card.json"), "sha256": "a" * 64},
                                  original_record=original,
                                  use="Synthetic software-only fast parameter input")
    monkeypatch.setattr(binder, "_construction_record", lambda case, family: deepcopy(old_inputs[family]))
    source_oracle = dict(scientific_base_commit=binder.SCIENTIFIC_BASE,
                         files_sha256={"synthetic-protected-source.py": "b" * 64},
                         binder_sha256=binder._hash(_PATH))
    monkeypatch.setattr(binder, "_source_binding", lambda case, family: deepcopy(source_oracle))
    return case, tmp_path


@pytest.fixture
def negative_record(synthetic):
    case, directory = synthetic
    record = binder.capture_record(case, "atlas", directory / "capture")
    assert record["status"] == "UNRESOLVED"
    assert record["observations"][0]["passed"] is False
    return record, case


@pytest.mark.parametrize("family", binder.FAMILIES)
def test_fresh_public_candidate_and_negative_replay(synthetic, family):
    case, directory = synthetic
    global_before = binder._fingerprint(binder._globals())
    record = binder.capture_record(case, family, directory / "capture")
    assert binder.verify_record(record, case, family) == record
    assert binder._fingerprint(binder._globals()) == global_before
    assert record["status"] == "UNRESOLVED" and record["observations"][0]["failed_bounds"]
    assert record["recipe_overrides"] == binder.SHARED_OVERRIDES
    assert record["resolved_recipe"]["d_lr_mult"] == 4.5
    assert record["resolved_recipe"]["lr"] == .00265625
    assert record["resolved_recipe"]["prior_lr_mult"] == 3.0
    state = torch.load(record["artifacts"]["state"]["path"], weights_only=True)
    api = state["api_state"]
    assert api["completed_steps"] == 0 and api["max_steps"] == 600
    assert api["initial_lrs"][0] == [.00265625, .00796875, .00265625]
    assert api["initial_lrs"][1] == [.011953125]
    assert api["initial_lrs"][0][0] == .5 * .0053125
    assert api["initial_lrs"][0][1] == .0053125 * 1.5
    assert api["initial_lrs"][1][0] == .0053125 * 2.25
    assert not any(optimizer["state"] for optimizer in api["optimizers"])
    assert binder.study._same_state(api["models"]["G"], api["models"]["ema_G"])
    assert binder.study._same_state(api["models"]["prior"], api["models"]["ema_prior"])
    assert record["construction"]["old_policy_optimizer_averages_clocks_transplanted"] is False
    assert record["observations"][0]["samples"] == 1024
    assert record["lifecycle"]["public_preludes"] == 1


@pytest.mark.parametrize("field,value", [
    ("recipe_overrides", {"lr": .0053125, "prior_lr_mult": 1.5}),
    ("requested_recipe_overrides", {"lr": .0053125, "prior_lr_mult": 1.5, "d_lr_mult": 1.5}),
    ("ordinary_training_updates", 1), ("ordinary_qualification_credit", True),
    ("family", "e22"), ("seed", 24003),
])
def test_reject_changed_declaration(negative_record, field, value):
    record, case = negative_record
    record[field] = value
    with pytest.raises(ValueError, match="binding differs"):
        binder.verify_record(record, case, "atlas")


def test_reject_stale_source_binding(negative_record):
    record, case = negative_record
    record["source"]["files_sha256"]["synthetic-protected-source.py"] = "c" * 64
    with pytest.raises(ValueError, match="binding differs"):
        binder.verify_record(record, case, "atlas")


def test_reject_changed_gate_even_with_coherent_case_hash(negative_record):
    record, case = negative_record
    changed = deepcopy(case)
    changed["thresholds"]["hq_min"] = .001
    record["case"] = changed
    record["bindings"]["case_sha256"] = binder.study.digest(changed)
    with pytest.raises(ValueError, match="gates.*changed"):
        binder.verify_record(record, changed, "atlas")


@pytest.mark.parametrize("change", ("clock", "horizon", "recipe", "optimizer", "model_nan", "ema", "scalar_type"))
def test_reject_coherently_rehashed_wrong_public_state(negative_record, change):
    record, case = negative_record
    path = Path(record["artifacts"]["state"]["path"])
    state = torch.load(path, weights_only=True)
    api = state["api_state"]
    if change == "clock":
        api["completed_steps"] = 1
    elif change == "horizon":
        api["max_steps"] = 601
    elif change == "recipe":
        api["recipe"]["d_lr_mult"] = 1.5
    elif change == "optimizer":
        api["optimizers"][1]["state"][0] = {"step": torch.tensor(1.)}
    elif change == "model_nan":
        next(iter(api["models"]["G"].values())).view(-1)[0] = float("nan")
    elif change == "ema":
        next(iter(api["models"]["ema_G"].values())).view(-1)[0] += 1
    else:
        assert api["controller"]["game_trust"] == 1.0
        api["controller"]["game_trust"] = 1
    torch.save(state, path)
    _rebind(record, "state")
    record["observer_purity"]["complete_public_state_sha256"] = binder._fingerprint(state)
    with pytest.raises(ValueError):
        binder.verify_record(record, case, "atlas")


@pytest.mark.parametrize("change", ("sample_count", "sample_value", "target", "extra_key", "nan", "signed_zero"))
def test_reject_coherently_rehashed_wrong_arrays(negative_record, change):
    record, case = negative_record
    path = Path(record["artifacts"]["samples"]["path"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if change == "sample_count":
        arrays["view0_samples"] = arrays["view0_samples"][:-1]
    elif change == "sample_value":
        arrays["view0_samples"].flat[0] += .01
    elif change == "target":
        arrays["view0_target"].flat[0] += .01
    elif change == "extra_key":
        arrays["unbound"] = np.zeros(1)
    elif change == "nan":
        arrays["view0_samples"].flat[0] = np.nan
    else:
        before = arrays["view0_target"].copy()
        first_zero = np.flatnonzero(before.reshape(-1) == 0)[0]
        arrays["view0_target"].flat[first_zero] = -0.0
        assert np.array_equal(arrays["view0_target"], before)
        assert arrays["view0_target"].tobytes() != before.tobytes()
    np.savez_compressed(path, **arrays)
    _rebind(record, "samples")
    with pytest.raises(ValueError, match="arrays differ"):
        binder.verify_record(record, case, "atlas")


@pytest.mark.parametrize("change", ("count", "seed", "metric", "false_pass", "status", "prelude", "backend", "transplant"))
def test_reject_forged_observation_or_lifecycle(negative_record, change):
    record, case = negative_record
    row = record["observations"][0]
    if change == "count":
        row["samples"] = 1000
    elif change == "seed":
        row["evaluation_seed"] += 1
    elif change == "metric":
        row["metrics"]["hq"] = .99
    elif change == "false_pass":
        row["passed"], row["failed_bounds"] = True, []
    elif change == "status":
        record["status"] = "SUPPORTED"
    elif change == "prelude":
        record["lifecycle"]["public_preludes"] = 0
    elif change == "backend":
        record["serving"]["source"] = "averaged"
    else:
        record["construction"]["old_policy_optimizer_averages_clocks_transplanted"] = True
    with pytest.raises(ValueError):
        binder.verify_record(record, case, "atlas")


def test_blocked_error_binds_failure_without_sampling(synthetic, monkeypatch):
    case, directory = synthetic
    record = binder.api_run.json_value(binder._base_record(case, "atlas"))
    problem = dict(stage="input_hash_check", type="ValueError", message="synthetic missing input", traceback="software-only")
    error = directory / "preparation-error.json"
    error.write_text(json.dumps(problem))
    record.update(status="BLOCKED", observations=[], artifacts={}, capacity_credit=False,
                  error=problem, error_artifact=dict(path=str(error), sha256=binder._hash(error), bytes=error.stat().st_size))
    monkeypatch.setattr(binder, "_construct", lambda *a: pytest.fail("blocked record cannot claim a sampler replay"))
    assert binder.verify_record(record, case, "atlas") == record
    record["error"]["message"] = "forged"
    with pytest.raises(ValueError, match="error differs"):
        binder.verify_record(record, case, "atlas")


def test_fingerprint_preserves_documented_nan_state_and_exact_tensor_bits():
    assert binder._fingerprint({"sentinel": float("nan")}) == binder._fingerprint({"sentinel": float("nan")})
    assert binder._fingerprint(torch.tensor([0.])) != binder._fingerprint(torch.tensor([-0.]))


def test_pinned_source_reads_exact_git_blob_without_history_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(binder, "ROOT", tmp_path)
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=b"fixed source bytes\n")
    monkeypatch.setattr(binder.subprocess, "run", run)
    binder._pinned_hash.cache_clear()
    assert binder._pinned_hash("particlegan/policy.py") == hashlib.sha256(b"fixed source bytes\n").hexdigest()
    assert calls == [(["git", "show", f"{binder.SCIENTIFIC_BASE}:particlegan/policy.py"],
                      {"cwd": tmp_path, "check": True, "capture_output": True})]
    binder._pinned_hash.cache_clear()


def test_reject_changed_physical_source_against_fixed_oracle(tmp_path, monkeypatch):
    monkeypatch.setattr(binder, "ROOT", tmp_path)
    monkeypatch.setattr(binder.contract, "ROOT", tmp_path)
    paths = ["particlegan/policy.py", "benchmarks/toy_audit/api_run.py", "benchmarks/toy_audit/api_family_search.py",
             "experiments/forge/contracts.py", "experiments/forge/boundaries.py", "experiments/forge/configuration_search.py"]
    for name in paths:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"oracle")
    proof = {"source_files_sha256": {paths[0]: hashlib.sha256(b"oracle").hexdigest()}}
    monkeypatch.setattr(binder.study, "proof_bindings", lambda *a: proof)
    monkeypatch.setattr(binder, "selected_cases", lambda: {"case": {"id": "case"}})
    monkeypatch.setattr(binder, "_pinned_hash", lambda name: hashlib.sha256(b"oracle").hexdigest())
    monkeypatch.setattr(binder.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="d" * 40))
    monkeypatch.setattr(binder, "_verify_committed_source", lambda source: None)
    monkeypatch.setattr(binder, "_check_import_origins", lambda: None)
    assert binder._source_binding({"id": "case"}, "atlas")["files_sha256"][paths[0]] == proof["source_files_sha256"][paths[0]]
    (tmp_path / paths[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="protected scientific source changed"):
        binder._source_binding({"id": "case"}, "atlas")


def test_require_all_sixteen_exact_outcomes_before_packet_verification(tmp_path, monkeypatch):
    packet = dict(schema=binder.SCHEMA, status="COMPLETE", records=[])
    path = tmp_path / "capacity.json"
    path.write_text(json.dumps(packet))
    monkeypatch.setattr(binder, "verify_record", lambda *a: pytest.fail("invalid denominator must fail before record admission"))
    with pytest.raises(ValueError, match="sixteen distinct"):
        binder.verify_packet(packet)
    assert binder.main(["--verify", str(path)]) == 2


@pytest.mark.parametrize("change", ("clock", "cuda", "horizon"))
def test_old_input_requires_actual_cpu_zero_clock_full_horizon(synthetic, change):
    case, _ = synthetic
    construction = binder._construction_record(case, "atlas")
    artifact = construction["original_record"]["artifacts"]["state"]
    path = Path(artifact["path"])
    state = torch.load(path, weights_only=True)
    api = state["api_state"]
    if change == "clock":
        api["completed_steps"] = 1
    elif change == "cuda":
        api["device"] = "cuda:0"
    else:
        api["max_steps"] += 1
    torch.save(state, path)
    artifact["sha256"] = binder._hash(path)
    with pytest.raises(ValueError, match="actual zero-update full-horizon CPU capture"):
        binder._input_state(construction, case)


def test_changed_original_input_hash_rejects_before_public_construction(synthetic, monkeypatch):
    case, directory = synthetic
    original = binder._construction_record(case, "atlas")
    path = Path(original["original_record"]["artifacts"]["state"]["path"])
    path.write_bytes(path.read_bytes() + b"tamper")
    monkeypatch.setattr(binder, "_construct", lambda *a: pytest.fail("changed input must fail before construction"))
    with pytest.raises(ValueError, match="construction artifact.*changed"):
        binder.capture_record(case, "atlas", directory / "capture")


def test_cpu_guard_rejects_cuda_initialized_parent_without_cuda_allocation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="no initialized CUDA"):
        binder._runtime()


def test_capacity_session_preserves_backend_flags_on_error():
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.benchmark, binder._fingerprint(binder._globals()))
    with pytest.raises(RuntimeError, match="controlled"):
        with binder._capacity_session():
            assert torch.are_deterministic_algorithms_enabled() and not torch.backends.cudnn.benchmark
            torch.rand(1)
            raise RuntimeError("controlled")
    after = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, binder._fingerprint(binder._globals()))
    assert after == before


def test_unattested_prerequisite_is_archived_and_never_a_zero_row_success(tmp_path, monkeypatch):
    cases = {name: {"id": name} for name in binder.CASE_IDS}
    monkeypatch.setattr(binder, "selected_cases", lambda: cases)
    def blocked(*args):
        raise ValueError("controlled changed source prerequisite")
    monkeypatch.setattr(binder, "_base_record", blocked)
    monkeypatch.setattr(binder, "capture_record", lambda *a: pytest.fail("source failure must block before construction"))
    directory = tmp_path / "blocked"
    assert binder.main(["--output", str(directory)]) == 2
    packet = json.loads((directory / "capacity.json").read_text())
    assert packet["status"] == "BLOCKED_PREREQUISITE" and packet["records"] == []
    assert len(packet["requested_cells"]) == packet["required_records"] == 16
    failure = packet["prerequisite_failure"]
    assert failure["qualification_credit"] is False
    path = binder._artifact(failure["artifact"])
    assert json.loads(path.read_text()) == failure["error"]
    with pytest.raises(ValueError, match="sixteen distinct"):
        binder.verify_packet(packet)


def test_committed_reproducer_and_every_scientific_blob_use_fixed_hash_oracle(monkeypatch):
    protected = "particlegan/policy.py"
    script = _PATH.relative_to(binder.ROOT).as_posix()
    oracle = {protected: "a" * 64, script: "b" * 64}
    calls = []
    def git_hash(commit, name):
        calls.append((commit, name))
        return oracle[name]
    monkeypatch.setattr(binder, "_git_blob_hash", git_hash)
    source = dict(reproducer_commit="c" * 40, files_sha256={protected: oracle[protected]}, binder_sha256=oracle[script])
    binder._verify_committed_source(source)
    assert set(calls) == {("c" * 40, protected), ("c" * 40, script)}
    source["binder_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="committed reproducer"):
        binder._verify_committed_source(source)
    source["reproducer_commit"] = "HEAD"
    with pytest.raises(ValueError, match="exact committed"):
        binder._verify_committed_source(source)


def test_cpu_cli_distinguishes_verified_negative_sibling_from_invalid_packet(tmp_path, monkeypatch, capsys):
    records = [dict(family=family, case_id=name, status="UNRESOLVED" if family == "e22" else "SUPPORTED")
               for family in binder.FAMILIES for name in binder.CASE_IDS]
    packet = dict(schema=binder.SCHEMA, status="COMPLETE", records=records)
    path = tmp_path / "capacity.json"
    path.write_text(json.dumps(packet))
    monkeypatch.setattr(binder, "verify_packet", lambda packet: packet)
    assert binder.main(["--verify", str(path)]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["verified"] is True and summary["packet_sha256"] == binder._hash(path)
    assert summary["family_supported"] == {"atlas": True, "e22": False}
    def invalid(packet):
        raise ValueError("controlled stale source")
    monkeypatch.setattr(binder, "verify_packet", invalid)
    assert binder.main(["--verify", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["verified"] is False


def test_cpu_guard_rejects_visible_gpu_before_public_cpu_fixture_forks_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="no visible CUDA"):
        binder._runtime()


def test_loaded_public_module_must_match_the_selected_physical_checkout(monkeypatch):
    binder._check_import_origins()
    monkeypatch.setitem(binder.sys.modules, "particlegan.capacity_rogue",
                        SimpleNamespace(__file__="/tmp/foreign-source/particlegan/capacity_rogue.py"))
    with pytest.raises(ValueError, match="different source"):
        binder._check_import_origins()


@pytest.mark.parametrize('family', binder.FAMILIES)
def test_all_eight_full_host_recipes_change_only_declared_transport_rates(family):
    """Pure metadata admission, no fixture, samples or parameter updates."""
    previous = {'lr': .0053125, 'prior_lr_mult': 1.5, 'd_lr_mult': 2.25}
    cases = binder.selected_cases()
    assert tuple(cases) == binder.CASE_IDS and len(cases) == 8
    for case in cases.values():
        old = binder.study.resolved_recipe(case, family, previous)
        new = binder.study.resolved_recipe(case, family, binder.SHARED_OVERRIDES)
        assert {key for key in old if old[key] != new[key]} == {'lr', 'prior_lr_mult', 'd_lr_mult'}
        assert new['lr'] == .5 * old['lr']
        assert new['lr'] * new['prior_lr_mult'] == old['lr'] * old['prior_lr_mult']
        assert new['lr'] * new['d_lr_mult'] == old['lr'] * old['d_lr_mult']
        assert new['total_steps'] is None
        assert case['default_steps'] in (600, 1200, 7000)


def test_original_critic_tuple_cannot_supply_new_capacity_identity(negative_record):
    record, case = negative_record
    record['recipe_overrides'] = {'lr': .0053125, 'prior_lr_mult': 1.5, 'd_lr_mult': 2.25}
    record['requested_recipe_overrides'] = deepcopy(record['recipe_overrides'])
    record['resolved_recipe'] = binder.study.resolved_recipe(case, 'atlas', record['recipe_overrides'])
    with pytest.raises(ValueError, match='binding differs'):
        binder.verify_record(record, case, 'atlas')
