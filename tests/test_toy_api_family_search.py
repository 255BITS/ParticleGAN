"""Software controls for the finite policy search; no scientific campaign."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from particlegan import GANTrainer, Recipe, get_recipe
from benchmarks.toy_audit import api_contract, api_run, api_family_search as search


@pytest.fixture(autouse=True)
def cpu_one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cases():
    found = api_contract.discover()
    return {name: found[name] for name, _ in search.DEFAULT_CASES}


def spec_for(tmp_path):
    return {"schema": search.SCHEMA, "id": "software-policy-study", "families": ["atlas", "e22"],
            "seed": 24002, "grid": {"lr": [.006375, .0085], "prior_lr_mult": [1., 2.]},
            "cases": [{"id": name, "tier": tier, "timeout_seconds": 10.} for name, tier in search.DEFAULT_CASES],
            "candidate_budget_seconds": 100., "budget_seconds": 1000., "export_grace_seconds": 1.,
            "stability": {"confirmation_checks": 5, "post_confirmation_hold_checks": 5,
                          "first_window_only": True, "all_subsequent_primary_checks": True},
            "backend": "cpu", "speed_ranking": False, "default_adoption": False, "frames": 3,
            "representation_card": {"path": str(tmp_path / "capacity.json"), "sha256": "not-installed"}}


def planned(tmp_path, cases, monkeypatch):
    monkeypatch.setattr(search, "_proofs", lambda spec, selected: {
        (family, name): {"status": "SUPPORTED", "reason": None}
        for family in search.FAMILIES for name in selected})
    return search.plan_study(spec_for(tmp_path), cases=cases)


@pytest.mark.parametrize("family", ["atlas", "e22"])
@pytest.mark.parametrize("identifier", [search.DEFAULT_CASES[0][0], "api-vector-two-broad"])
@pytest.mark.parametrize("knobs", [{"lr": .006375, "prior_lr_mult": 1.},
                                   {"lr": .002125, "prior_lr_mult": 1.},
                                   {"lr": .00425, "prior_lr_mult": 2.},
                                   {"lr": .0031875, "prior_lr_mult": 1.},
                                   {"lr": .0053125, "prior_lr_mult": 2.}])
def test_actual_public_recipe_knobs_applied_before_optimizer_and_updated(cases, family, identifier, knobs):
    fixture = api_contract.build(cases[identifier], device="cpu", recipe_name=family,
                                 max_steps=2, recipe_overrides=knobs)
    assert isinstance(fixture.recipe, Recipe)
    assert fixture.recipe.lr == knobs["lr"] and fixture.recipe.prior_lr_mult == knobs["prior_lr_mult"]
    assert api_run.json_value(fixture.recipe.to_dict()) == search.resolved_recipe(cases[identifier], family, knobs)
    assert fixture.trainer.opt_g.param_groups[0]["lr"] == knobs["lr"]
    before = [parameter.detach().clone() for parameter in fixture.trainer.G.parameters()]
    fixture.step(); fixture.step()
    assert fixture.trainer.completed_steps == 2
    assert any(not torch.equal(a, b) for a, b in zip(before, fixture.trainer.G.parameters()))
    observed = api_contract.validate_observation(fixture.observe(n=32, seed=34002))
    assert observed["metrics"] and observed["views"]
    assert fixture.state_dict()["recipe"]["lr"] == knobs["lr"]
    search._check_health(fixture.state_dict(), cases[identifier], fixture.recipe.to_dict(), completed_steps=2)


@pytest.mark.parametrize("override", [{"z_dim": 1}, {"prior_kind": "mog"}, {"num_particles": 1},
                                      {"continuous_policy": None}, {"lr": float("nan")}])
def test_host_sampler_and_invalid_public_fields_rejected_before_optimizers(cases, monkeypatch, override):
    monkeypatch.setattr(Recipe, "make_optimizers", lambda *args, **kwargs: pytest.fail("optimizer was constructed"))
    with pytest.raises((ValueError, TypeError)):
        api_contract.build(cases["api-vector-two-broad"], recipe_name="atlas", recipe_overrides=override)


@pytest.mark.parametrize("definition", [
    {"id": "conditional", "provider": "api_conditionals"},
    {"id": "diagnostic", "provider": "api_diagnostics"},
    {"id": "paired", "provider": "api_images", "query": "mask"},
    {"id": "word", "provider": "api_images", "kind": "word"},
    {"id": "cadence", "provider": "api_vectors", "caller_owned": True},
    {"id": "api-stress-r1-r2", "provider": "api_vectors", "penalty_arm": "a_r1r2"},
    {"id": "api-stress-fast-critic", "provider": "api_vectors"},
])
def test_unsupported_context_control_knobs_blocked(definition):
    with pytest.raises(api_contract.UnsupportedRecipeOverrides):
        api_contract.validate_recipe_overrides(definition, "atlas", {"lr": .006375})


def timing_receipt(flags):
    return {"protocol": {"metric_evaluation_steps": list(range(len(flags) + 1))},
            "observations": [{"step": i, "passed": passed, "elapsed_seconds": i / 10}
                             for i, passed in enumerate([True, *flags])]}


def test_first_confirmation_then_full_hold_has_bound_time():
    result = search.acquisition_hold(timing_receipt([False, True] + [True] * 10))
    assert result["status"] == "PASS" and result["acquired_step"] == 6
    assert result["acquired_seconds"] == .6 and result["hold_checks"] == 6
    assert result["speed_eligible"] is False


@pytest.mark.parametrize("flags,status", [([True]*9, "INCOMPLETE"), ([False]*24, "FAIL"),
                                          ([True]*5+[False]+[True]*18, "FAIL")])
def test_missing_hold_or_late_collapse_cannot_be_selected(flags, status):
    assert search.acquisition_hold(timing_receipt(flags))["status"] == status


def test_initial_and_media_only_successes_do_not_count():
    receipt = timing_receipt([False] * 12)
    receipt["observations"].insert(2, {"step": .5, "passed": True, "elapsed_seconds": .15})
    assert search.acquisition_hold(receipt)["status"] == "FAIL"
    receipt["observations"].pop(5)
    assert search.acquisition_hold(receipt)["status"] == "INCOMPLETE"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1., True])
def test_invalid_acquisition_time_cannot_rank(value):
    receipt = timing_receipt([True] * 12)
    receipt["observations"][6]["elapsed_seconds"] = value
    with pytest.raises(ValueError, match="timing"):
        search.acquisition_hold(receipt)


def test_nonmonotonic_acquisition_time_rejected():
    receipt = timing_receipt([True] * 12)
    receipt["observations"][6]["elapsed_seconds"] = .1
    with pytest.raises(ValueError, match="timing"):
        search.acquisition_hold(receipt)


def test_all_configs_and_full_denominator_retained_no_speed_winner(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    assert len(packet["trials"]) == 8
    assert {trial["family"] for trial in packet["trials"]} == {"atlas", "e22"}
    assert all(len(trial["cases"]) == 8 for trial in packet["trials"])
    assert search.select_results(packet)["outcome"] == "pending"
    for trial in packet["trials"]:
        trial["status"] = "FAIL"
        trial["cases"][0]["status"] = "FAIL"
    for trial in packet["trials"][:2]:
        trial["status"] = "PASS"
        for row in trial["cases"]:
            row["status"] = "PASS"
    result = search.select_results(packet)
    assert result["outcome"] == "scoped_fully_qualified"
    assert len(result["fully_qualified_ids"]) == 2 and result["speed_winner"] is None
    assert result["required_cases_per_config"] == 8 and result["default_adoption"] is False


@pytest.mark.parametrize("learning_rates", [[.006375, .0085], [.002125, .00425], [.0031875, .0053125]])
def test_frozen_grid_profiles_have_distinct_complete_config_denominators(tmp_path, cases, monkeypatch, learning_rates):
    spec = spec_for(tmp_path)
    spec["grid"]["lr"] = learning_rates
    monkeypatch.setattr(search, "_proofs", lambda spec, selected: {
        (family, name): {"status": "SUPPORTED", "reason": None}
        for family in search.FAMILIES for name in selected})
    packet = search.plan_study(spec, cases=cases)
    assert len(packet["trials"]) == 8
    assert {(trial["recipe_overrides"]["lr"], trial["recipe_overrides"]["prior_lr_mult"])
            for trial in packet["trials"]} == {(lr, rate) for lr in learning_rates for rate in (1., 2.)}
    assert all(len(trial["cases"]) == 8 for trial in packet["trials"])
    assert search.select_results(packet)["outcome"] == "pending"
    for other_rates in search.LR_PROFILES:
        if list(other_rates) == learning_rates:
            continue
        packet["spec"]["grid"]["lr"] = list(other_rates)
        with pytest.raises(ValueError, match="configuration"):
            search.select_results(packet)  # Old IDs cannot fill any different grid.


@pytest.mark.parametrize("grid", [{"lr": [.00425, .00425], "prior_lr_mult": [1., 2.]},
                                  {"lr": [.002125, .006375], "prior_lr_mult": [1., 2.]},
                                  {"lr": [.0031875, .00425], "prior_lr_mult": [1., 2.]},
                                  {"lr": [.0031875, .0053125, .006375], "prior_lr_mult": [1., 2.]},
                                  {"lr": [.003, .005], "prior_lr_mult": [1., 2.]},
                                  {"lr": [.002125, .00425], "prior_lr_mult": [1., 1.]},
                                  {"lr": [.002125, .00425], "prior_lr_mult": [True, 2.]},
                                  {"lr": [.002125, float("nan")], "prior_lr_mult": [1., 2.]}])
def test_custom_duplicate_or_malformed_grid_cannot_change_search_scope(tmp_path, cases, grid):
    spec = spec_for(tmp_path)
    spec["grid"] = grid
    with pytest.raises(ValueError, match="grid"):
        search.validate_spec(spec, cases)


@pytest.mark.parametrize("mutation", ["drop_config", "duplicate_config", "drop_case", "duplicate_case", "pass_unknown"])
def test_incomplete_conflicting_scopes_cannot_shrink_denominator(tmp_path, cases, monkeypatch, mutation):
    packet = planned(tmp_path, cases, monkeypatch)
    if mutation == "drop_config":
        packet["trials"].pop()
    elif mutation == "duplicate_config":
        packet["trials"][0] = deepcopy(packet["trials"][1])
    elif mutation == "drop_case":
        packet["trials"][0]["cases"].pop()
    elif mutation == "duplicate_case":
        packet["trials"][0]["cases"][1] = deepcopy(packet["trials"][0]["cases"][0])
    else:
        packet["trials"][0]["status"] = "PASS"
    with pytest.raises(ValueError):
        search.select_results(packet)


def test_budget_and_gate_changes_are_not_implicit_search_axes(tmp_path, cases):
    spec = spec_for(tmp_path)
    assert search.validate_spec(spec, cases) is spec
    for changed in (dict(spec, candidate_budget_seconds=1), dict(spec, budget_seconds=1),
                    dict(spec, speed_ranking=True), dict(spec, seed=123),
                    dict(spec, grid={"lr": [.00425], "prior_lr_mult": [2.]}),
                    dict(spec, stability={"confirmation_checks": 1})):
        with pytest.raises(ValueError):
            search.validate_spec(changed, cases)


@pytest.mark.parametrize("protocol_seed", [24499, "24002", 24002.])
def test_case_cli_seed_must_equal_frozen_study_before_capacity_or_child(tmp_path, cases, monkeypatch, protocol_seed):
    changed = deepcopy(cases)
    changed[search.DEFAULT_CASES[-1][0]]["protocol_seed"] = protocol_seed
    monkeypatch.setattr(search, "_proofs", lambda *args: pytest.fail("seed drift reached capacity admission"))
    with pytest.raises(ValueError, match="protocol seed"):
        search.plan_study(spec_for(tmp_path), cases=changed)


@pytest.mark.parametrize("identifier", [name for name, _ in search.DEFAULT_CASES])
@pytest.mark.parametrize("family", search.FAMILIES)
def test_generated_child_arguments_parse_through_actual_api_main_without_updates(tmp_path, cases, monkeypatch, identifier, family):
    spec = spec_for(tmp_path)
    search.validate_spec(spec, cases)
    trial = {"family": family, "recipe_overrides": {"lr": .006375, "prior_lr_mult": 1.}}
    row = next(item for item in spec["cases"] if item["id"] == identifier)
    case_root = tmp_path / family / identifier
    command = search.child_command(spec, trial, row, case_root, "cpu")
    requests = []
    def execute(request):
        requests.append(request)
        return {"id": identifier, "status": "INCOMPLETE", "verdict": "FAIL"}
    # Exercise the real parser/discovery/request construction, replacing only
    # paid execution. A --help probe would miss the rejected legacy --seed.
    monkeypatch.setattr(api_run, "_execute_request", execute)
    assert api_run.main(command[3:]) == 1
    assert len(requests) == 1
    actual_case, output, options = requests[0]
    assert actual_case == cases[identifier] and output == case_root
    assert options == {"device": "cpu", "recipe_name": family, "steps": None,
                       "eval_samples": None, "frames": spec["frames"],
                       "recipe_overrides": trial["recipe_overrides"],
                       "wall_cap_seconds": row["timeout_seconds"], "seed": spec["seed"]}
    assert "--seed" not in command and not case_root.exists()
    requests.clear()
    with pytest.raises(SystemExit) as rejected:
        api_run.main([*command[3:], "--seed", "24002"])
    assert rejected.value.code == 2 and not requests


def test_health_checks_learned_state_without_rejecting_controller_sentinels():
    case = {"default_steps": 12}
    recipe = {"name": "atlas"}
    state = {"trainer": {"completed_steps": 12, "recipe": recipe,
                         "models": {"G": {"weight": torch.ones(2)}}, "optimizers": [{"state": {}}],
                         "controller": {"invalid_history": torch.tensor(float("nan"))}}}
    search._check_health(state, case, recipe)
    state["trainer"]["models"]["G"]["weight"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        search._check_health(state, case, recipe)


def test_forged_capacity_sample_metrics_or_horizon_rejected(tmp_path, monkeypatch):
    case = {"id": "software-capacity", "provider": "api_vectors", "default_steps": 12, "eval_samples": 4}
    state_path, samples_path = tmp_path / "state.pt", tmp_path / "samples.npz"
    state = {"trainer": {"recipe": {"name": "atlas"}, "completed_steps": 0, "max_steps": 12,
                         "models": {"G": {"weight": torch.ones(2)}}, "optimizers": [{"state": {}}]}}
    torch.save(state, state_path)
    np.savez(samples_path, samples=np.zeros((4, 2)))
    monkeypatch.setattr(search, "_score", lambda *args: {"passed": True, "failed_bounds": [], "metrics": {"error": 0.}})
    monkeypatch.setattr(search, "resolved_recipe", lambda *args: {"name": "atlas"})
    monkeypatch.setattr(search, "_replay_capacity", lambda *args: None)
    record = {"observations": [{"passed": True, "failed_bounds": [], "metrics": {"error": 0.}}],
              "artifacts": {"state": {"path": str(state_path)}, "samples": {"path": str(samples_path)}}}
    search._check_capacity(record, case, "atlas")
    record["observations"][0]["metrics"]["error"] = 1.
    with pytest.raises(ValueError, match="metrics"):
        search._check_capacity(record, case, "atlas")
    record["observations"][0]["metrics"]["error"] = 0.
    state["trainer"]["max_steps"] = 1
    torch.save(state, state_path)
    with pytest.raises(ValueError, match="horizon"):
        search._check_capacity(record, case, "atlas")


def test_missing_capacity_is_blocked_not_omitted(tmp_path, cases):
    spec = spec_for(tmp_path)
    card = Path(spec["representation_card"]["path"])
    api_run.write_json(card, {"records": []})
    spec["representation_card"]["sha256"] = api_run.file_hash(card)
    packet = search.plan_study(spec, cases=cases)
    assert all(trial["status"] == "BLOCKED" and len(trial["cases"]) == 8 for trial in packet["trials"])
    assert search.select_results(packet)["outcome"] == "incomplete_comparison"
    card.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        search.plan_study(spec, cases=cases)


def test_source_guarded_union_uses_only_owned_family_results(tmp_path, cases, monkeypatch):
    original = planned(tmp_path, cases, monkeypatch)
    paths = []
    for family in search.FAMILIES:
        packet = deepcopy(original)
        packet.update(executed_family=family, lane_runtime=search._runtime("cpu"), spent_seconds=0.,
                      measured_paid_seconds=0., unmeasured_interrupt_reservation_seconds=0.)
        for trial in packet["trials"]:
            if trial["family"] == family:
                trial["status"] = "INCOMPLETE"
                trial["cases"][0]["status"] = "INCOMPLETE"
        path = tmp_path / (family + ".json")
        api_run.write_json(path, packet); paths.append(path)
    result = search.combine_studies(paths)
    assert result["spent_seconds"] == 0. and result["selection"]["attempts_concluded"]
    assert not result["selection"]["comparison_complete"]
    assert all(row["status"] == "UNKNOWN" for trial in result["trials"] for row in trial["cases"][1:])
    changed = json.loads(paths[1].read_text())
    changed["source"]["commit"] = "different-cohort"
    api_run.write_json(paths[1], changed)
    with pytest.raises(ValueError, match="cohorts"):
        search.combine_studies(paths)


def test_run_freezes_before_child_keeps_full_resources_and_stops_first_failure(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    monkeypatch.setattr(search, "plan_study", lambda spec: deepcopy(packet))
    monkeypatch.setattr(search, "_source", lambda selected: deepcopy(packet["source"]))
    archive = tmp_path / "run"
    commands = []
    def child(command, **kwargs):
        frozen = json.loads((archive / "study.json").read_text())
        assert frozen["source"] == packet["source"]
        assert sum(row["status"] == "RUNNING" for trial in frozen["trials"] for row in trial["cases"]) == 1
        assert "--steps" not in command and "--eval-samples" not in command and "--workers" not in command
        commands.append(command)
        output = Path(command[command.index("--output") + 1]) / command[command.index("--case") + 1]
        api_run.write_json(output / "receipt.json", {"status": "INCOMPLETE", "verdict": "FAIL", "failed_bounds": ["cap"]})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(search.subprocess, "run", child)
    result = search.run_study(packet["spec"], archive, family="atlas", device="cpu")
    assert len(commands) == 4 and all("--seed" not in command for command in commands)
    atlas = [trial for trial in result["trials"] if trial["family"] == "atlas"]
    assert all(trial["status"] == "INCOMPLETE" and trial["cases"][0]["status"] == "INCOMPLETE" for trial in atlas)
    assert all(row["status"] == "UNKNOWN" for trial in atlas for row in trial["cases"][1:])
    search.run_study(packet["spec"], archive, family="atlas", device="cpu")
    assert len(commands) == 4  # No unchanged failed or capped retry.
    with pytest.raises(ValueError, match="separate"):
        search.run_study(packet["spec"], archive, family="e22", device="cpu")


def test_interrupted_paid_attempt_is_not_automatically_reexecuted(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    monkeypatch.setattr(search, "plan_study", lambda spec: deepcopy(packet))
    monkeypatch.setattr(search, "_source", lambda selected: deepcopy(packet["source"]))
    packet.update(executed_family="atlas", lane_runtime=search._runtime("cpu"), spent_seconds=0.)
    trial = next(trial for trial in packet["trials"] if trial["family"] == "atlas")
    trial["cases"][0]["status"] = "RUNNING"
    archive = tmp_path / "run"
    api_run.write_json(archive / "study.json", packet)
    monkeypatch.setattr(search.subprocess, "run", lambda *args, **kwargs: pytest.fail("unchanged interrupted attempt reran"))
    # Other fresh configs are blocked in this software fixture so only recovery is evaluated.
    for candidate in packet["trials"]:
        if candidate["id"] != trial["id"]:
            candidate["status"] = "BLOCKED"
    api_run.write_json(archive / "study.json", packet)
    result = search.run_study(packet["spec"], archive, family="atlas", device="cpu")
    recovered = next(candidate for candidate in result["trials"] if candidate["id"] == trial["id"])
    assert recovered["status"] == "INCOMPLETE" and result["spent_seconds"] == 11.
    assert recovered["cases"][0]["unmeasured_interrupt_reserved_seconds"] == 11.


class PublicTimedFixture:
    """Real public optimizer updates; finite output is a software-only gate."""
    api_components = ("GANTrainer", "Recipe")
    def __init__(self, clock=None, max_steps=12):
        self.recipe = get_recipe("atlas", z_dim=2, num_particles=16, batch_size=16,
                                 lr=.006375, prior_lr_mult=1.)
        critic = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 1))
        self.trainer = GANTrainer(self.recipe, nn.Linear(2, 2), critic, seed=7, max_steps=max_steps)
        self.data = torch.Generator().manual_seed(9)
        self.clock = clock
    def step(self):
        self.trainer.step(torch.randn(16, 2, generator=self.data))
        if self.clock is not None:
            self.clock[0] += .4
    def observe(self, n, seed):
        samples = self.trainer.sample(n, generator=torch.Generator().manual_seed(seed), output_noise=False)
        return {"metrics": {"finite_fraction": float(torch.isfinite(samples).float().mean())},
                "passed": True, "failed_bounds": [],
                "views": [{"kind": "scatter", "title": "Actual software-control outputs", "target": torch.zeros(n, 2), "samples": samples}]}
    def state_dict(self):
        return {"recipe": self.recipe.to_dict(), "trainer": self.trainer.state_dict()}


def software_case():
    return {"id": "software-fresh-study", "provider": "api_vectors", "title": "Software fresh-study gate",
            "goal": "Exercise public API and strict evidence verifier", "kind": "vector", "scope": "software control, no trained distribution claim",
            "default_steps": 12, "eval_samples": 4, "batch_size": 16, "legacy_ids": ["software-control"],
            "thresholds": [["finite_fraction", "==", 1.]], "sampling": "actual public served software-control draws",
            "evaluation_observations": 12}


@pytest.fixture
def fresh_software_run(tmp_path, monkeypatch):
    fixture = PublicTimedFixture()
    monkeypatch.setattr(api_contract, "build", lambda *args, **kwargs: fixture)
    case = software_case()
    path = tmp_path / "fresh"
    receipt = api_run.run_case(case, path, recipe_name="atlas", recipe_overrides={"lr": .006375, "prior_lr_mult": 1.},
                               frames=3, wall_cap_seconds=100.)
    assert receipt["verdict"] == "PASS"
    monkeypatch.setattr(search, "_score", lambda *args: {"metrics": {"finite_fraction": 1.}, "passed": True, "failed_bounds": []})
    monkeypatch.setattr(search, "resolved_recipe", lambda *args: api_run.json_value(fixture.recipe.to_dict()))
    return path, case, receipt


def test_complete_fresh_source_bound_api_gate_and_hold(fresh_software_run):
    path, case, receipt = fresh_software_run
    result = search.verify_case(path, case, "atlas", {"lr": .006375, "prior_lr_mult": 1.}, receipt["source"],
                                returncode=0, runtime=receipt["runtime"], wall_cap_seconds=100., frames=3)
    assert result["original_gate"] == result["study_gate"] == result["status"] == "PASS"
    assert result["acquisition_hold"]["hold_checks"] == 7


@pytest.mark.parametrize("mutation", ["source", "recipe", "seed", "runtime", "exit", "cap", "partial", "metric", "weights"])
def test_invalid_fresh_evidence_cannot_be_certified(fresh_software_run, mutation):
    path, case, receipt = fresh_software_run
    source, runtime = deepcopy(receipt["source"]), deepcopy(receipt["runtime"])
    code = 0
    if mutation == "source":
        source["commit"] = "different-source"
    elif mutation == "recipe":
        receipt["recipe"]["lr"] *= 2
    elif mutation == "seed":
        receipt["seed"] = 1
    elif mutation == "runtime":
        runtime["python"] = "different-runtime"
    elif mutation == "exit":
        code = 1
    elif mutation == "cap":
        receipt["protocol"]["wall_cap_seconds"] = .000001
    elif mutation == "partial":
        receipt["completed_updates"] -= 1
    elif mutation == "metric":
        receipt["observations"][0]["metrics"]["finite_fraction"] = .5
    elif mutation == "weights":
        state = torch.load(path / "final-state.pt", weights_only=True)
        next(iter(state["trainer"]["models"]["G"].values())).view(-1)[0] = float("nan")
        torch.save(state, path / "final-state.pt")
        receipt["artifacts"]["final-state.pt"] = {"sha256": api_run.file_hash(path / "final-state.pt"), "bytes": (path / "final-state.pt").stat().st_size}
    api_run.write_json(path / "receipt.json", receipt)
    with pytest.raises(ValueError):
        search.verify_case(path, case, "atlas", {"lr": .006375, "prior_lr_mult": 1.}, source, returncode=code,
                           runtime=runtime, wall_cap_seconds=100., frames=3)


def test_actual_cap_retains_partial_state_without_extra_update(tmp_path, monkeypatch):
    clock = [0.]
    fixture = PublicTimedFixture(clock)
    monkeypatch.setattr(api_contract, "build", lambda *args, **kwargs: fixture)
    monkeypatch.setattr(api_run.time, "monotonic", lambda: clock[0])
    receipt = api_run.run_case(software_case(), tmp_path / "capped", frames=3, wall_cap_seconds=1.)
    assert receipt["status"] == "INCOMPLETE" and receipt["verdict"] == "FAIL"
    assert receipt["completed_updates"] == fixture.trainer.completed_steps == 3
    assert receipt["protocol"]["updates"] == 12 and not receipt["default_protocol_complete"]
    assert receipt["observations"][-1]["step"] == 3
    assert {"observations.npz", "final-state.pt", "goal.gif"} == set(receipt["artifacts"])
    with pytest.raises(ValueError, match="unsuccessful"):
        search.api_publish.verify_run(tmp_path / "capped")


def test_main_failure_returns_nonzero_instead_of_default_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(search, "run_study", lambda *args, **kwargs: {"selection": {"fully_qualified_ids": []}})
    api_run.write_json(tmp_path / "spec.json", {})
    assert search.main(["run", str(tmp_path / "spec.json"), "--family", "atlas", "--output", str(tmp_path / "out")]) == 1


def test_unrelated_oracle_samples_cannot_prove_zero_generator_capacity(tmp_path, cases):
    from benchmarks.toy_audit import api_vectors
    case = cases["api-vector-two-broad"]
    fixture = api_contract.build(case, recipe_name="e22", max_steps=case["default_steps"])
    with torch.no_grad():
        for parameter in fixture.trainer.G.parameters():
            parameter.zero_()
    state_path, samples_path = tmp_path / "state.pt", tmp_path / "samples.npz"
    torch.save(fixture.state_dict(), state_path)
    oracle = api_vectors._target(case, case["eval_samples"], torch.Generator().manual_seed(713),
                                 case["default_steps"])
    score = api_vectors.score_case(case, oracle, 0)
    assert score["passed"]  # Unrelated clean arrays would satisfy the original gate.
    np.savez(samples_path, samples=oracle.numpy())
    record = {"observations": [score], "artifacts": {"state": {"path": str(state_path)}, "samples": {"path": str(samples_path)}}}
    with pytest.raises(ValueError, match="restored public sampler"):
        search._check_capacity(record, case, "e22")


def test_capacity_replay_checks_primary_gate_without_requiring_extra_clean_diagnostics(tmp_path, cases):
    # A small software draw through the actual native host; no optimizer
    # updates or scientific capacity/convergence claim is made here.
    case = {**cases["api-grid100"], "eval_samples": 64}
    fixture = api_contract.build(case, recipe_name="e22", seed=24002,
                                 max_steps=case["default_steps"])
    observed = fixture.observe(n=case["eval_samples"], seed=34002)
    samples = api_contract.array(observed["views"][0]["samples"])
    primary = search._score(case, samples, 0)
    assert "clean_gate_passed" in observed["metrics"]
    assert "clean_gate_passed" not in primary["metrics"]
    record = {"observations": [{**primary, "evaluation_seed": 34002}]}
    state = fixture.state_dict()
    search._replay_capacity(record, case, "e22", state, {"samples": samples})
    changed = deepcopy(record)
    metric = next(iter(primary["metrics"]))
    changed["observations"][0]["metrics"][metric] += 1.
    with pytest.raises(ValueError, match="restored public sampler"):
        search._replay_capacity(changed, case, "e22", state, {"samples": samples})
    with pytest.raises(ValueError, match="restored public sampler"):
        search._replay_capacity(record, case, "e22", state, {"samples": samples + 1.})
    # Ordinary receipts retain the additional diagnostics and canonical sorted
    # failure ordering; their original primary values must still be checked.
    row = {"step": 0, **api_contract.validate_observation(observed)}
    np.savez(tmp_path / "observations.npz", step0_view0_samples=samples)
    search._check_numeric_trace(tmp_path, {"observations": [row]}, case)
    row["metrics"][metric] += 1.
    with pytest.raises(ValueError, match="recorded numeric gate"):
        search._check_numeric_trace(tmp_path, {"observations": [row]}, case)


def test_closed_incomplete_study_cannot_name_final_comparison_winner(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    for trial in packet["trials"]:
        trial["status"] = "INCOMPLETE"
    trial = packet["trials"][0]
    trial["status"] = "PASS"
    for row in trial["cases"]:
        row["status"] = "PASS"
    result = search.select_results(packet)
    assert result["attempts_concluded"] and not result["comparison_complete"]
    assert result["outcome"] == "incomplete_comparison" and result["fully_qualified_ids"] == [trial["id"]]
    assert result["speed_winner"] is None


def test_family_quotas_do_not_double_round_allowance(tmp_path, cases):
    spec = spec_for(tmp_path)
    spec["budget_seconds"] = 100.
    spec["candidate_budget_seconds"] = 100.
    search.validate_spec(spec, cases)  # Worst-case704 s may exceed bounded paid100 s.
    spec["family_budget_seconds"] = {"atlas": 60., "e22": 50.}
    with pytest.raises(ValueError, match="round paid cap"):
        search.validate_spec(spec, cases)


def test_complete_numeric_failure_is_retained_with_exit_one(fresh_software_run, monkeypatch):
    path, case, receipt = fresh_software_run
    for row in receipt["observations"]:
        row.update(passed=False, failed_bounds=["software rejection"])
    receipt.update(passed=False, verdict="FAIL", metric_passed=False, sustained_metric_passed=False,
                   failed_bounds=["software rejection", "last 5 post-update metric observations do not all pass"])
    monkeypatch.setattr(search, "_score", lambda *args: {"metrics": {"finite_fraction": 1.}, "passed": False, "failed_bounds": ["software rejection"]})
    api_run.write_json(path / "receipt.json", receipt)
    result = search.verify_case(path, case, "atlas", {"lr": .006375, "prior_lr_mult": 1.}, receipt["source"],
                                returncode=1, runtime=receipt["runtime"], wall_cap_seconds=100., frames=3)
    assert result["original_gate"] == result["study_gate"] == result["status"] == "FAIL"
    with pytest.raises(ValueError, match="exit"):
        search.verify_case(path, case, "atlas", {"lr": .006375, "prior_lr_mult": 1.}, receipt["source"],
                           returncode=0, runtime=receipt["runtime"], wall_cap_seconds=100., frames=3)


def test_terminal_archive_pass_needs_recertified_receipts_and_costs(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    packet.update(executed_family="atlas", lane_runtime=search._runtime("cpu"), spent_seconds=1.,
                  measured_paid_seconds=1., unmeasured_interrupt_reservation_seconds=0.)
    trial = next(trial for trial in packet["trials"] if trial["family"] == "atlas")
    trial["status"] = "FAIL"
    trial["paid_wall_seconds"] = 1.
    row = trial["cases"][0]
    row.update(status="FAIL", paid_wall_seconds=1., original_gate="FAIL", study_gate="FAIL", full_protocol_complete=True,
               acquisition_hold={"status": "FAIL"}, final_metrics={"software": 1.}, recipe={"name": "atlas"},
               runtime=packet["lane_runtime"], artifacts={}, child_returncode=1)
    receipt_path = tmp_path / "retained-receipt.json"
    api_run.write_json(receipt_path, {"software": "self-contained archived receipt control"})
    row.update(receipt_path=str(receipt_path), receipt_sha256=api_run.file_hash(receipt_path))
    calls = []
    def verify(*args, **kwargs):
        calls.append(kwargs)
        return {**deepcopy(row), "elapsed_seconds": .5}
    monkeypatch.setattr(search, "verify_case", verify)
    search._recertify_archive(packet)
    assert len(calls) == 1 and calls[0]["wall_cap_seconds"] == 10. and calls[0]["frames"] == 3
    row["status"] = "PASS"
    monkeypatch.setattr(search, "verify_case", lambda *args, **kwargs: {**row, "status": "FAIL", "elapsed_seconds": .5})
    with pytest.raises(ValueError, match="scientific status"):
        search._recertify_archive(packet)
    receipt_path.write_text("changed")
    with pytest.raises(ValueError, match="unchanged bound receipt"):
        search._recertify_archive(packet)


@pytest.mark.parametrize("mutation", ["negative", "reduced_total", "reduced_trial", "nonfinite"])
def test_saved_budget_cannot_admit_more_paid_work(tmp_path, cases, monkeypatch, mutation):
    packet = planned(tmp_path, cases, monkeypatch)
    packet.update(spent_seconds=5., measured_paid_seconds=5., unmeasured_interrupt_reservation_seconds=0.)
    packet["trials"][0]["paid_wall_seconds"] = 5.
    packet["trials"][0]["cases"][0]["paid_wall_seconds"] = 5.
    search._verify_costs(packet)
    if mutation == "negative":
        packet["trials"][0]["cases"][0]["paid_wall_seconds"] = -1.
    elif mutation == "reduced_total":
        packet["spent_seconds"] = 0.
    elif mutation == "reduced_trial":
        packet["trials"][0]["paid_wall_seconds"] = 0.
    else:
        packet["measured_paid_seconds"] = float("nan")
    with pytest.raises(ValueError, match="cost"):
        search._verify_costs(packet)


def test_archived_metadata_cannot_lower_frozen_gates(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    packet.update(executed_family="atlas", lane_runtime=search._runtime("cpu"), spent_seconds=0.,
                  measured_paid_seconds=0., unmeasured_interrupt_reservation_seconds=0.)
    search._recertify_archive(packet)
    packet["case_definitions"]["api-vector-two-broad"]["thresholds"][-1][-1] = 1.
    with pytest.raises(ValueError, match="frozen registry"):
        search._recertify_archive(packet)


@pytest.mark.parametrize("mutation", ["quota", "timeout", "interrupt_reservation"])
def test_mutable_allowances_cannot_enlarge_preregistered_budget(tmp_path, cases, monkeypatch, mutation):
    packet = planned(tmp_path, cases, monkeypatch)
    packet.update(executed_family="atlas", lane_runtime=search._runtime("cpu"), spent_seconds=0.,
                  measured_paid_seconds=0., unmeasured_interrupt_reservation_seconds=0.)
    if mutation == "quota":
        packet["family_paid_budget_seconds"]["atlas"] += 100.
    elif mutation == "timeout":
        packet["trials"][0]["cases"][0]["timeout_seconds"] += 100.
    else:
        packet["trials"][0]["cases"][0]["unmeasured_interrupt_reserved_seconds"] = 0.
    with pytest.raises(ValueError, match="preregistration|allowance"):
        search._recertify_archive(packet)


def test_coherent_zeroed_costs_still_cannot_erase_actual_acquisition(tmp_path, cases, monkeypatch):
    packet = planned(tmp_path, cases, monkeypatch)
    packet.update(executed_family="atlas", lane_runtime=search._runtime("cpu"), spent_seconds=0.,
                  measured_paid_seconds=0., unmeasured_interrupt_reservation_seconds=0.)
    trial = next(trial for trial in packet["trials"] if trial["family"] == "atlas")
    trial["status"] = "FAIL"
    row = trial["cases"][0]
    receipt = tmp_path / "receipt.json"
    api_run.write_json(receipt, {"software": "recorded positive acquisition duration"})
    row.update(status="FAIL", receipt_path=str(receipt), receipt_sha256=api_run.file_hash(receipt), paid_wall_seconds=0.)
    # Aggregate, candidate and row zeros are mutually consistent, but cannot
    # erase a nonzero duration independently bound by the executed receipt.
    search._verify_costs(packet)
    monkeypatch.setattr(search, "verify_case", lambda *args, **kwargs: {"elapsed_seconds": .5})
    with pytest.raises(ValueError, match="acquisition time"):
        search._recertify_archive(packet)


def test_real_original_pass_with_insufficient_hold_still_binds_receipt_and_paid_cost(tmp_path, cases, monkeypatch):
    # Actual public updates and retained arrays/state/GIF, with a deliberately
    # narrow finite-output software predicate. Nine primary checks provide an
    # original PASS but only four checks after the five-check confirmation.
    identifier = search.DEFAULT_CASES[0][0]
    case = {**software_case(), "id": identifier, "default_steps": 9, "evaluation_observations": 9,
            "z_dim": 2, "particles": 16}
    catalog = {**cases, identifier: case}
    monkeypatch.setattr(api_contract, "discover", lambda: catalog)
    fixture = PublicTimedFixture(max_steps=9)
    monkeypatch.setattr(api_contract, "build", lambda *args, **kwargs: fixture)
    monkeypatch.setattr(search, "_score", lambda *args: {
        "metrics": {"finite_fraction": 1.}, "passed": True, "failed_bounds": []})
    packet = planned(tmp_path, catalog, monkeypatch)
    trial = next(trial for trial in packet["trials"] if trial["family"] == "atlas"
                 and trial["recipe_overrides"] == {"lr": .006375, "prior_lr_mult": 1.})
    path = tmp_path / "actual-complete-short-hold"
    receipt = api_run.run_case(case, path, recipe_name="atlas", recipe_overrides=trial["recipe_overrides"],
                               frames=3, wall_cap_seconds=10.)
    assert receipt["verdict"] == "PASS" and receipt["default_protocol_complete"]
    verified = search.verify_case(path, case, "atlas", trial["recipe_overrides"], packet["source"],
                                  returncode=0, runtime=receipt["runtime"], wall_cap_seconds=10., frames=3)
    assert verified["status"] == verified["study_gate"] == "INCOMPLETE"
    assert verified["original_gate"] == "PASS" and verified["acquisition_hold"]["hold_checks"] == 4
    paid = receipt["elapsed_seconds"] + .5
    row = trial["cases"][0]
    row.update(verified, paid_wall_seconds=paid, child_returncode=0)
    trial.update(status="INCOMPLETE", paid_wall_seconds=paid)
    packet.update(executed_family="atlas", lane_runtime=receipt["runtime"], spent_seconds=paid,
                  measured_paid_seconds=paid, unmeasured_interrupt_reservation_seconds=0.)
    search._recertify_archive(packet)  # Real retained-original-PASS receipt; no mocked verifier.
    erased = deepcopy(packet)
    erased_trial = next(item for item in erased["trials"] if item["id"] == trial["id"])
    erased_trial["cases"][0]["paid_wall_seconds"] = erased_trial["paid_wall_seconds"] = 0.
    erased["spent_seconds"] = erased["measured_paid_seconds"] = 0.
    search._verify_costs(erased)  # Coherently forged zeros cannot evade the raw time.
    with pytest.raises(ValueError, match="acquisition time"):
        search._recertify_archive(erased)
    row["full_protocol_complete"] = False
    with pytest.raises(ValueError, match="scientific status"):
        search._recertify_archive(packet)
    row["full_protocol_complete"] = True
    row["receipt_sha256"] = "changed"
    with pytest.raises(ValueError, match="unchanged bound receipt"):
        search._recertify_archive(packet)
