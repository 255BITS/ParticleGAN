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
_SPEC = importlib.util.spec_from_file_location("ka2_k3p_prior2_capacity", _PATH)
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
            fixture = binder.contract.build(case, device="cpu", seed=binder.SEED, recipe_name="atlas",
                                             max_steps=case["default_steps"],
                                             recipe_overrides={"lr": .0053125, "prior_lr_mult": 1.5})
            state = fixture.state_dict()
        # Deliberately unrelated old averages: they must never be imported.
        for tensor in state["api_state"]["models"]["ema_G"].values():
            tensor.fill_(17)
        path = tmp_path / f"{family}-synthetic-input.pt"
        torch.save(state, path)
        original = dict(family="atlas", case_id=case["id"],
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
    record = binder.capture_record(case, "ka2", directory / "capture")
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
    assert record["resolved_recipe"]["d_lr_mult"] == 1.0
    assert record["resolved_recipe"]["lr"] == .006375
    assert record["resolved_recipe"]["prior_lr_mult"] == 2.0
    state = torch.load(record["artifacts"]["state"]["path"], weights_only=True)
    api = state["api_state"]
    assert api["completed_steps"] == 0 and api.get("max_steps", api["recipe"]["total_steps"]) == 600
    assert api["initial_lrs"][0] == [.006375, .01275]
    assert api["initial_lrs"][1] == [.006375]
    assert api["optimizers"][0]["param_groups"][0]["lr"] == .006375
    assert api["optimizers"][0]["param_groups"][1]["lr"] == .01275
    assert api["optimizers"][1]["param_groups"][0]["lr"] == .006375
    assert api.get("controller") is None and api["recipe"]["total_steps"] == 600
    assert api.get("output_noise") is None
    assert record["serving"]["source"] == "fast"
    assert record["serving"]["output_sigma"] == 0.
    assert record["serving"]["latent_perturbation_enabled"] is False
    assert not any(optimizer["state"] for optimizer in api["optimizers"])
    assert binder.study._same_state(api["models"]["G"], api["models"]["ema_G"])
    assert binder.study._same_state(api["models"]["prior"], api["models"]["ema_prior"])
    assert record["construction"]["old_policy_optimizer_averages_clocks_transplanted"] is False
    assert record["observations"][0]["samples"] == 1024
    assert record["lifecycle"]["public_preludes"] == 1


@pytest.mark.parametrize("field,value", [
    ("schema", "particlegan_ka2_k3p_capacity_v1"),
    ("recipe_overrides", {"lr": .006375, "prior_lr_mult": 1.0, "d_lr_mult": 1.0}),
    ("requested_recipe_overrides", {"lr": .006375, "prior_lr_mult": 1.0, "d_lr_mult": 1.0}),
    ("recipe_overrides", {"lr": .0053125, "prior_lr_mult": 1.5}),
    ("requested_recipe_overrides", {"lr": .0053125, "prior_lr_mult": 1.5, "d_lr_mult": 1.5}),
    ("ordinary_training_updates", 1), ("ordinary_qualification_credit", True),
    ("family", "k3p"), ("seed", 24003),
])
def test_reject_changed_declaration(negative_record, field, value):
    record, case = negative_record
    record[field] = value
    with pytest.raises(ValueError, match="binding differs"):
        binder.verify_record(record, case, "ka2")


@pytest.mark.parametrize("change", ("tuple", "family", "typed_prior"))
def test_sibling_protocol_cannot_silently_revert_or_widen_fixed_prior2_contract(monkeypatch, change):
    case = binder.selected_cases()["image-develop-img_intensity2-source-transpose12"]
    if change == "family":
        monkeypatch.setattr(binder.protocol, "FAMILIES", ("atlas", "e22"))
    else:
        overrides = deepcopy(binder.protocol.OVERRIDES)
        overrides["prior_lr_mult"] = 1.0 if change == "tuple" else 2
        monkeypatch.setattr(binder.protocol, "OVERRIDES", overrides)
    with pytest.raises(ValueError, match="prior2 protocol.*exact declared"):
        binder._canonical(case, "ka2")


@pytest.mark.parametrize("change", ("prior_group", "initial_prior"))
def test_coherently_rehashed_prior_one_optimizer_ownership_is_not_new_prior2_capacity(negative_record, change):
    record, case = negative_record
    path = Path(record["artifacts"]["state"]["path"])
    state = torch.load(path, weights_only=True)
    # Keep the new Recipe/request labels, but transplant one old rate field.
    # Exact public construction must detect this despite coherent file hashes.
    if change == "prior_group":
        state["api_state"]["optimizers"][0]["param_groups"][1]["lr"] = .006375
    else:
        state["api_state"]["initial_lrs"][0][1] = .006375
    torch.save(state, path)
    _rebind(record, "state")
    record["observer_purity"]["complete_public_state_sha256"] = binder._fingerprint(state)
    with pytest.raises(ValueError):
        binder.verify_record(record, case, "ka2")


def test_reject_stale_source_binding(negative_record):
    record, case = negative_record
    record["source"]["files_sha256"]["synthetic-protected-source.py"] = "c" * 64
    with pytest.raises(ValueError, match="binding differs"):
        binder.verify_record(record, case, "ka2")


def test_reject_changed_gate_even_with_coherent_case_hash(negative_record):
    record, case = negative_record
    changed = deepcopy(case)
    changed["thresholds"]["hq_min"] = .001
    record["case"] = changed
    record["bindings"]["case_sha256"] = binder.study.digest(changed)
    with pytest.raises(ValueError, match="gates.*changed"):
        binder.verify_record(record, changed, "ka2")


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
        api["recipe"]["serve_average"] = 0
    torch.save(state, path)
    _rebind(record, "state")
    record["observer_purity"]["complete_public_state_sha256"] = binder._fingerprint(state)
    with pytest.raises(ValueError):
        binder.verify_record(record, case, "ka2")


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
        binder.verify_record(record, case, "ka2")


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
        binder.verify_record(record, case, "ka2")


def test_blocked_error_binds_failure_without_sampling(synthetic, monkeypatch):
    case, directory = synthetic
    record = binder.api_run.json_value(binder._base_record(case, "ka2"))
    problem = dict(stage="input_hash_check", type="ValueError", message="synthetic missing input", traceback="software-only")
    error = directory / "preparation-error.json"
    error.write_text(json.dumps(problem))
    record.update(status="BLOCKED", observations=[], artifacts={}, capacity_credit=False,
                  error=problem, error_artifact=dict(path=str(error), sha256=binder._hash(error), bytes=error.stat().st_size))
    monkeypatch.setattr(binder, "_construct", lambda *a: pytest.fail("blocked record cannot claim a sampler replay"))
    assert binder.verify_record(record, case, "ka2") == record
    record["error"]["message"] = "forged"
    with pytest.raises(ValueError, match="error differs"):
        binder.verify_record(record, case, "ka2")


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
    protocol_path = tmp_path / "reports/new/protocol.py"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_bytes(b"fixed new protocol")
    monkeypatch.setattr(binder, "_PROTOCOL_PATH", protocol_path)
    paths = ["configs/forge/tasks/ring16_acquisition.json", "particlegan/policy.py", "benchmarks/toy_audit/api_run.py", "benchmarks/toy_audit/api_family_search.py",
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
    assert binder._source_binding({"id": "case"}, "ka2")["files_sha256"][paths[0]] == proof["source_files_sha256"][paths[0]]
    (tmp_path / paths[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="protected scientific source changed"):
        binder._source_binding({"id": "case"}, "ka2")


def test_require_all_sixteen_exact_outcomes_before_packet_verification(tmp_path, monkeypatch):
    packet = dict(schema=binder.SCHEMA, status="COMPLETE", records=[])
    path = tmp_path / "capacity.json"
    path.write_text(json.dumps(packet))
    monkeypatch.setattr(binder, "verify_record", lambda *a: pytest.fail("invalid denominator must fail before record admission"))
    with pytest.raises(ValueError, match="sixteen distinct"):
        binder.verify_packet(packet)
    assert binder.main(["--verify", str(path)]) == 2


@pytest.mark.parametrize("change", ("missing", "duplicate", "unknown", "old_schema"))
def test_full_synthetic_packet_cannot_borrow_or_drop_a_declared_capacity_cell(monkeypatch, change):
    records = [dict(family=family, case_id=name) for family in binder.FAMILIES for name in binder.CASE_IDS]
    packet = dict(schema=binder.SCHEMA, status="COMPLETE", claim_scope=binder.CLAIM_SCOPE,
                  required_records=16, records=records,
                  requested_cells=[dict(family=family, case_id=name)
                                   for family in binder.FAMILIES for name in binder.CASE_IDS],
                  ordinary_training_updates=0, fitting_updates=0, ordinary_qualification_credit=False)
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = deepcopy(records[0])
    elif change == "unknown":
        records[-1]["case_id"] = "api-new-unregistered-host"
    else:
        packet["schema"] = "particlegan_ka2_k3p_capacity_v1"
    monkeypatch.setattr(binder, "verify_record", lambda *a: pytest.fail("bad denominator/schema must reject before replay"))
    with pytest.raises(ValueError, match="sixteen distinct"):
        binder.verify_packet(packet)


@pytest.mark.parametrize("change", ("clock", "cuda", "horizon"))
def test_old_input_requires_actual_cpu_zero_clock_full_horizon(synthetic, change):
    case, _ = synthetic
    construction = binder._construction_record(case, "ka2")
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
    original = binder._construction_record(case, "ka2")
    path = Path(original["original_record"]["artifacts"]["state"]["path"])
    path.write_bytes(path.read_bytes() + b"tamper")
    monkeypatch.setattr(binder, "_construct", lambda *a: pytest.fail("changed input must fail before construction"))
    with pytest.raises(ValueError, match="construction artifact.*changed"):
        binder.capture_record(case, "ka2", directory / "capture")


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
    protocol = binder._PROTOCOL_PATH.relative_to(binder.ROOT).as_posix()
    oracle = {protected: "a" * 64, script: "b" * 64, protocol: "e" * 64}
    calls = []
    def git_hash(commit, name):
        calls.append((commit, name))
        return oracle[name]
    monkeypatch.setattr(binder, "_git_blob_hash", git_hash)
    source = dict(reproducer_commit="c" * 40, files_sha256={protected: oracle[protected]},
                  new_test_files_sha256={protocol: oracle[protocol]}, binder_sha256=oracle[script])
    binder._verify_committed_source(source)
    assert set(calls) == {("c" * 40, protected), ("c" * 40, script), ("c" * 40, protocol)}
    source["binder_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="committed reproducer"):
        binder._verify_committed_source(source)
    source["reproducer_commit"] = "HEAD"
    with pytest.raises(ValueError, match="exact committed"):
        binder._verify_committed_source(source)


def test_cpu_cli_distinguishes_verified_negative_sibling_from_invalid_packet(tmp_path, monkeypatch, capsys):
    records = [dict(family=family, case_id=name, status="UNRESOLVED" if family == "k3p" else "SUPPORTED")
               for family in binder.FAMILIES for name in binder.CASE_IDS]
    packet = dict(schema=binder.SCHEMA, status="COMPLETE", records=records)
    path = tmp_path / "capacity.json"
    path.write_text(json.dumps(packet))
    monkeypatch.setattr(binder, "verify_packet", lambda packet: packet)
    assert binder.main(["--verify", str(path)]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["verified"] is True and summary["packet_sha256"] == binder._hash(path)
    assert summary["family_supported"] == {"ka2": True, "k3p": False}
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
def test_all_eight_actual_family_recipes_bind_factory_horizon(family):
    cases = binder.selected_cases()
    assert tuple(cases) == binder.CASE_IDS and len(cases) == 8
    for case in cases.values():
        recipe = binder.protocol.resolved_recipe(case, family)
        assert recipe['lr'] == .006375 and recipe['prior_lr_mult'] == 2.
        assert recipe['d_lr_mult'] == 1. and recipe['total_steps'] == case['default_steps']
        assert recipe['continuous_policy'] is None and recipe['serve_average'] == 0.
        assert recipe['output_noise_mode'] == 'fixed' and recipe['output_noise_warmup'] == .2
        assert recipe['amsgrad'] is False
        assert case['default_steps'] in (600, 1200, 7000)
        law = binder.protocol.family_law(case, family)
        assert law['critic_formulation'] == family and law['served_model'] == 'fast_only'
        assert law['actual_read_standardize'] is False


def test_coordinate_construction_preserves_original_public_mlp_scaffold_and_map():
    from lib.toy_models import SimpleMLPGenerator
    # Small software-only host retains the original public architecture class.
    with binder.api_run.isolated_evaluation():
        torch.manual_seed(1)
        generator = SimpleMLPGenerator(4, 8, 3, 2)
    objects = [id(module) for module in generator.net]
    parameters = [id(parameter) for parameter in generator.parameters()]
    binder._coordinate_identity(generator)
    points = torch.tensor([[0., 0., 7., -3.], [2., -4., 10., 15.], [-2., 4., -7., 8.],
                           [-.01, -.005, 0., 0.], [1., 1., 2., 2.]])
    torch.testing.assert_close(generator(points), points[:, :2], atol=1e-6, rtol=1e-6)
    assert [id(module) for module in generator.net] == objects
    assert [id(parameter) for parameter in generator.parameters()] == parameters


def test_wrong_coordinate_host_cannot_be_replaced_with_identity():
    with pytest.raises(ValueError, match='original sequential MLP'):
        binder._coordinate_identity(torch.nn.Linear(4, 2))
    wrong = SimpleNamespace(net=torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)))
    with pytest.raises(ValueError, match='original Linear/LeakyReLU'):
        binder._coordinate_identity(wrong)


def test_unequal_occupancy_keeps_all_rare_rows_without_mass_substitution():
    counts = binder._component_counts([.55, .30, .13, .02], 256)
    assert counts.tolist() == [141, 77, 33, 5] and int(counts.sum()) == 256
    counts = binder._component_counts([1/3, 1/3, 1/3], 256)
    assert counts.tolist() == [86, 85, 85]
    with pytest.raises(ValueError, match='at least four'):
        binder._component_counts([.99, .01], 16)


def test_tiny_finite_population_preserves_full_signed_covariance_without_draws():
    spec = dict(kind='gaussian_mixture', masses=[.5, .5], means=[[-1., .5], [1., -.5]],
                covariances=[[[.09, .018], [.018, .0081]], [[.0081, -.018], [-.018, .09]]])
    case = dict(spec=spec, particles=32, z_dim=4)
    globals_before = binder._fingerprint(binder._globals())
    rows = binder._vector_population(case).numpy()
    assert rows.shape == (32, 4) and np.count_nonzero(rows[:, 2:]) == 0
    for index in range(2):
        group = rows[index * 16:(index + 1) * 16, :2]
        np.testing.assert_allclose(group.mean(0), spec['means'][index], atol=1e-6)
        np.testing.assert_allclose(np.cov(group, rowvar=False, bias=True), spec['covariances'][index], atol=1e-7)
    assert binder._fingerprint(binder._globals()) == globals_before


def test_tiny_radial_design_has_antipodal_width_without_randomness():
    globals_before = binder._fingerprint(binder._globals())
    offsets = binder._radial_pairs(8, .03)
    assert offsets.shape == (16, 2)
    np.testing.assert_array_equal(offsets[:8], -offsets[8:])
    radii = np.linalg.norm(offsets[:8], axis=1)
    np.testing.assert_allclose(1 - np.exp(-radii ** 2 / (2 * .03 ** 2)), (np.arange(8) + .5) / 8)
    assert (radii > 0).all() and np.linalg.eigvalsh(np.cov(offsets, rowvar=False, bias=True)).min() > 0
    assert binder._fingerprint(binder._globals()) == globals_before


@pytest.mark.parametrize('family', binder.FAMILIES)
def test_tiny_public_family_sampler_has_no_perturbation_or_warm_sigma_at_zero(family):
    from particlegan import GANTrainer, get_recipe
    rows = torch.tensor([[-1., 0.], [-.9, .02], [1., 0.], [.9, -.02]])
    recipe = get_recipe(family, z_dim=2, num_particles=4, batch_size=4, total_steps=20)
    with binder.api_run.isolated_evaluation():
        torch.manual_seed(12)
        prior = recipe.make_prior()
        prior.z.data.copy_(rows)
        generator = torch.nn.Linear(2, 2)
        generator.weight.data.copy_(torch.eye(2)); generator.bias.data.zero_()
        critic = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.LeakyReLU(.2), torch.nn.Linear(4, 1))
        trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=12, max_steps=20)
        before = binder._fingerprint(trainer.state_dict())
        sample = trainer.sample(64, generator=torch.Generator().manual_seed(5), output_noise=True)
        assert all(any(torch.equal(point, row) for row in rows) for point in sample)
        assert trainer.policy.controller is None and trainer.log_output_sigma is None
        assert trainer.output_sigma() == 0 and trainer.completed_steps == 0
        assert binder._fingerprint(trainer.state_dict()) == before
    # This four-row software law is not a native fidelity certificate.


@pytest.mark.parametrize('change', ('integer_metric', 'integer_rate', 'false_zero_flag'))
def test_typed_binding_and_metrics_reject_equal_value_forgery(negative_record, change):
    record, case = negative_record
    if change == 'integer_metric':
        record['observations'][0]['metrics']['hq'] = 0
    elif change == 'integer_rate':
        record['recipe_overrides']['d_lr_mult'] = 1
    else:
        record['ordinary_training_updates'] = False
    with pytest.raises(ValueError):
        binder.verify_record(record, case, 'ka2')


def test_source_input_family_is_explicit_atlas_not_a_fabricated_ka2_lookup(monkeypatch):
    case = binder.selected_cases()['image-develop-img_intensity2-source-transpose12']
    oracle = dict(family='atlas', case_id=case['id'], bindings={'case_sha256': binder.study.digest(case)}, artifacts={})
    monkeypatch.setattr(binder, '_hash', lambda path: binder.INPUT_CARD_SHA256)
    monkeypatch.setattr(Path, 'read_text', lambda self: json.dumps({'records': [oracle]}))
    for family in binder.FAMILIES:
        construction = binder._construction_record(case, family)
        assert construction['original_record'] == oracle
        assert 'No old noise' in construction['use']
