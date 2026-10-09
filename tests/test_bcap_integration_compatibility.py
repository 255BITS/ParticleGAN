"""Independent, exact inactive-path comparisons to the frozen develop source.

This is bounded software evidence, not a distribution-quality experiment.
The same probe runs in separate interpreters against two real source trees;
expected updates are never reconstructed from the candidate implementation.
"""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


DEVELOP = "5737ade47dca89b04d338ada20667781d1f2b5df"
ROOT = Path(os.environ.get("BCAP_COMPATIBILITY_ROOT", Path(__file__).resolve().parents[1]))
CASES = ("bcap_adam", "bcap_winner", "k3p", "ka2", "r1r2", "released_gan")
PRIORS = ("particle_cloud", "mog")
NEW_INACTIVE_FIELDS = {
    "constraint_geometry_mode": "none",
    "kinetic_transport_weight": 0.0,
    "kinetic_transport_local_weight": 0.0,
    "kinetic_transport_projections": 32,
}
WINNER = "bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36"
HOST_CARDS = {
    "bcap_adam": "bcap-pure-adam-v2",
    "bcap_winner": WINNER,
    "k3p": "k3p-global-repair-v1",
    "ka2": "ka2-global-repair-v1",
    "r1r2": "r1r2-global-repair-v1",
    "released_gan": "release07-gan-v3-task-adapted-v1",
    "e22": "e22",
    "atlas": "atlas",
}
COMPONENT_CASES = tuple((case, host) for case in CASES for host in ("trajectory", "ae_gan_hold")) + tuple(
    (case, host) for case in ("bcap_winner", "ka2")
    for host in ("two_pole", "unused_token_hold", "mid_scale_identity", "residual_student", "unipolar", "cover_leftover"))


def _assert_equal(left, right, path="state"):
    import torch
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), (path, left.keys() ^ right.keys())
        for key in left:
            _assert_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, (path, left, right)


def _card(root, name):
    folder = "configurations" if name == WINNER else "ideas"
    return json.loads((root / f"configs/forge/{folder}/{name}.json").read_text())


def _probe(root, output, baseline=None):
    # All production imports resolve against the requested source, independently
    # of the checkout containing this common, implementation-neutral probe.
    sys.path.insert(0, str(root))
    import torch
    from torch import nn
    from experiments.forge.api import FormulationContext
    from experiments.forge.adapters import adapter_preflight

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    packets = {}
    baseline_packets = torch.load(baseline, weights_only=False) if baseline else None

    def build(case, prior_kind):
        torch.manual_seed(0)
        if case == "released_gan":
            card = _card(root, HOST_CARDS[case])
            preset, overrides = card.get("recipe_preset"), dict(card["recipe_overrides"])
        elif case == "bcap_winner":
            card = _card(root, WINNER)
            preset, overrides = card["recipe_preset"], dict(card["recipe_overrides"])
        else:
            preset = "bcap_adam" if case == "bcap_adam" else "k3p" if case == "r1r2" else case
            overrides = {"reg_arm": "a_r1r2"} if case == "r1r2" else {}
        overrides.update(num_particles=12, z_dim=2, batch_size=6, total_steps=8)
        context = FormulationContext(recipe_preset=preset, recipe_overrides=overrides,
            prior={"kind": prior_kind, "sigma": .025 if prior_kind == "mog" else 0.,
                   "standardize": False, "learnable": True,
                   **({"exception_reason": "Separate historical particle-prior compatibility cohort"}
                      if prior_kind == "particle_cloud" else {})}, seed=0, device="cpu")
        generator = context.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(),
            nn.Dropout(.2), nn.Linear(8, 2)), component="generator")
        critic = context.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2),
            nn.Dropout(.1), nn.Linear(8, 1)), component="discriminator")
        trainer = context.build_trainer(generator, critic)
        # Disabled capabilities must never need a protected-loss binder, a
        # special optimizer, or either sample-space objective.
        assert not hasattr(trainer.opt_g, "bind_protected_losses")
        if hasattr(context.recipe, "constraint_geometry_mode"):
            assert context.recipe.constraint_geometry_mode == "none"
            assert context.recipe.kinetic_transport_weight == 0
            assert context.recipe.kinetic_transport_local_weight == 0
            def forbidden(*args, **kwargs):
                raise AssertionError("inactive transport was consumed")
            from particlegan import Recipe
            Recipe.kinetic_transport_loss = forbidden
            Recipe.kinetic_transport_local_loss = forbidden
        context.streams.generator("data", component="target", purpose="critic")
        context.streams.generator("data", component="target", purpose="generator")
        return context, trainer

    def updates(context, trainer, count):
        rows = []
        for _ in range(count):
            real_d = torch.randn((6, 2), generator=context.streams.generator(
                "data", component="target", purpose="critic"))
            real_g = torch.randn((6, 2), generator=context.streams.generator(
                "data", component="target", purpose="generator"))
            losses = trainer.step(real_d, generator_real=real_g)
            before = trainer.state_dict()
            global_before = torch.get_rng_state().clone()
            sample = trainer.sample(11, output_noise=True)
            after = trainer.state_dict()
            assert torch.equal(global_before, torch.get_rng_state())
            for name, state in before["streams"].items():
                if name != "eval_generator":
                    assert torch.equal(state, after["streams"][name]), name
            rows.append({"step": trainer.completed_steps, "real_d": real_d,
                         "real_g": real_g, "losses": losses, "sample": sample})
        return rows

    for case in CASES:
        for prior_kind in PRIORS:
            key = f"{case}/{prior_kind}"
            context, trainer = build(case, prior_kind)
            initial = {"trainer": trainer.state_dict(), "named": context.streams.state_dict()}
            prefix = updates(context, trainer, 3)
            checkpoint = {"trainer": trainer.state_dict(), "named": context.streams.state_dict()}
            suffix = updates(context, trainer, 3)
            final = {"trainer": trainer.state_dict(), "named": context.streams.state_dict()}
            packet = {"initial": initial, "checkpoint": checkpoint, "prefix": prefix,
                      "suffix": suffix, "final": final}
            if baseline_packets is not None:
                restored = []
                # Load the actual develop checkpoint, with absent optional
                # fields; also accept explicit inactive defaults and its older
                # schema-4 variant without the optional policy metadata.
                for variant in ("original", "explicit_inactive", "without_policy"):
                    context2, trainer2 = build(case, prior_kind)
                    saved = deepcopy(baseline_packets[key]["checkpoint"])
                    if variant == "explicit_inactive":
                        saved["trainer"]["recipe"].update(NEW_INACTIVE_FIELDS)
                    elif variant == "without_policy":
                        saved["trainer"].pop("policy", None)
                    context2.streams.load_state_dict(saved["named"])
                    trainer2.load_state_dict(saved["trainer"])
                    continued = updates(context2, trainer2, 3)
                    restored.append({"variant": variant, "suffix": continued,
                        "final": {"trainer": trainer2.state_dict(), "named": context2.streams.state_dict()}})
                packet["restored"] = restored
            packets[key] = packet
            print(json.dumps({"event": "compatibility_case_complete", "case": key}), flush=True)

    # Preserve original tasks and cards. A preexisting unsupported cohort must
    # remain explicit and must not acquire a projection/transport requirement.
    view = json.loads((root / "configs/forge/views/discriminator_stability.json").read_text())
    matrix = {}
    without_hooks = {}
    declared_tasks = {}
    for case, name in HOST_CARDS.items():
        card = _card(root, name)
        matrix[case] = {}
        without_hooks[case] = {}
        for assignment in view["assignments"]:
            if assignment["importance"] != "required" or assignment["qualification_tier"] > 2:
                continue
            task_id = assignment["task"]
            task = json.loads((root / f"configs/forge/tasks/{task_id}.json").read_text())
            declared_tasks[task_id] = task
            blockers = adapter_preflight(task, card, root=root)
            matrix[case][task_id] = blockers
            optional_absent = deepcopy(task)
            optional_absent["execution"].pop("transport_consumer", None)
            optional_absent["execution"].pop("transport_contract", None)
            without_hooks[case][task_id] = adapter_preflight(optional_absent, card, root=root)
    packets["host_matrix"] = matrix
    packets["without_hooks"] = without_hooks
    packets["declared_tasks"] = declared_tasks
    from experiments.forge.behavior_adapters import run_behavior
    components = {}
    for case, host in COMPONENT_CASES:
        # Separate two-update software cohort: original model/data/objective,
        # public host loop and explicit reduced budget. No quality verdict or
        # ordinary evidence is produced or imported by these checks.
        task = json.loads((root / f"configs/forge/tasks/{host}.json").read_text())
        task["execution"]["steps"] = 2
        directory = output.parent / f"{output.stem}-components" / case / host
        request = {"candidate": _card(root, HOST_CARDS[case]), "protocol": {"seed": 0}}
        run_behavior(request, task, directory, device="cpu")
        state = torch.load(directory / "component-state.pt", weights_only=False)
        # Public provenance added explicit inactive optional metadata. Its
        # value must be inactive; substantive training state remains exact.
        assert state.pop("component_transport", None) is None
        assert state["applied"].pop("component_transport", None) is None
        recipe = state["applied"]["recipe"]
        for name, default in NEW_INACTIVE_FIELDS.items():
            assert recipe.pop(name, default) == default
            ownership = state["applied"]["field_ownership"]["recipe_fields"].pop(name, None)
            if ownership is not None:
                assert ownership["value"] == default
                assert ownership["source"].startswith("public Recipe ")
        components[f"{case}/{host}"] = state
    packets["components"] = components
    from experiments.forge.word_adapter import word_context
    from benchmarks.toy_audit.api_images import WordFixture
    words = {}
    def build_words(case):
        torch.manual_seed(0)
        task = json.loads((root / "configs/forge/tasks/five_word_joint_smoke.json").read_text())
        context = word_context({"candidate": _card(root, HOST_CARDS[case]), "protocol": {"seed": 0}}, task, "cpu")
        fixture = WordFixture(device="cpu", seed=0, recipe_name=None, max_steps=4, components=context)
        return context, fixture
    for case in CASES:
        context, fixture = build_words(case)
        with torch.autograd.set_multithreading_enabled(False):
            prefix = [fixture.step() for _ in range(2)]
            checkpoint = {"fixture": fixture.state_dict(), "named": context.streams.state_dict()}
            suffix = [fixture.step() for _ in range(2)]
        final = {"fixture": fixture.state_dict(), "named": context.streams.state_dict()}
        packet = {"prefix": prefix, "checkpoint": checkpoint, "suffix": suffix, "final": final}
        if baseline_packets is not None:
            context2, fixture2 = build_words(case)
            saved = baseline_packets["words"][case]["checkpoint"]
            context2.streams.load_state_dict(saved["named"])
            fixture2.policy.load_state_dict(saved["fixture"]["api_state"])
            fixture2.data_generator.set_state(saved["fixture"]["data_generator"])
            with torch.autograd.set_multithreading_enabled(False):
                continued = [fixture2.step() for _ in range(2)]
            packet["restored"] = {"suffix": continued, "final": {
                "fixture": fixture2.state_dict(), "named": context2.streams.state_dict()}}
        words[case] = packet
    packets["words"] = words
    torch.save(packets, output)


if __name__ == "__main__":
    _probe(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
else:
    import pytest
    import torch

    @pytest.fixture(scope="module")
    def comparison(tmp_path_factory):
        directory = tmp_path_factory.mktemp("bcap-compatibility")
        reference = directory / "develop"
        reference.mkdir()
        try:
            archive = subprocess.run(["git", "archive", DEVELOP, "particlegan", "experiments",
                "benchmarks", "configs", "lib", "reports/transfer_suite/unadjusted/leading_profile.json"],
                cwd=ROOT, check=True, capture_output=True).stdout
        except subprocess.CalledProcessError as error:
            pytest.fail(f"Frozen develop source {DEVELOP} is required: {error.stderr.decode()}")
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(reference, filter="data")
        outputs = [directory / "develop.pt", directory / "candidate.pt"]
        env = {**os.environ, "PYTHONPATH": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        for index, source in enumerate((reference, ROOT)):
            command = [sys.executable, str(Path(__file__).resolve()), str(source), str(outputs[index])]
            if index:
                command.append(str(outputs[0]))
            log = directory / ("develop.log" if not index else "candidate.log")
            with log.open("w") as stdout:
                result = subprocess.run(command, cwd=source, env=env, stdout=stdout,
                                        stderr=subprocess.STDOUT, text=True, timeout=180)
            assert result.returncode == 0, log.read_text()
        return tuple(torch.load(output, weights_only=False) for output in outputs)

    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("prior_kind", PRIORS)
    def test_inactive_trainer_is_bitwise_develop(comparison, case, prior_kind):
        reference, candidate = comparison
        key = f"{case}/{prior_kind}"
        _assert_equal(reference[key], {name: value for name, value in candidate[key].items()
                                     if name != "restored"}, key)

    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("prior_kind", PRIORS)
    def test_actual_develop_checkpoint_resumes_exactly(comparison, case, prior_kind):
        reference, candidate = comparison
        key = f"{case}/{prior_kind}"
        assert not set(NEW_INACTIVE_FIELDS) & set(reference[key]["checkpoint"]["trainer"]["recipe"])
        for resumed in candidate[key]["restored"]:
            _assert_equal(reference[key]["suffix"], resumed["suffix"], key + resumed["variant"])
            _assert_equal(reference[key]["final"], resumed["final"], key + resumed["variant"])

    @pytest.mark.parametrize("case", tuple(HOST_CARDS))
    def test_original_host_acceptance_has_no_new_capability_blockers(comparison, case):
        reference, candidate = comparison
        assert len(reference["host_matrix"][case]) == 27
        _assert_equal(reference["host_matrix"][case], candidate["host_matrix"][case], case)
        _assert_equal(reference["host_matrix"][case], candidate["without_hooks"][case], case + ".without-hooks")
        for blockers in candidate["host_matrix"][case].values():
            assert not any("transport" in reason or "constraint_geometry" in reason for reason in blockers)
        if case in ("e22", "atlas"):
            assert all(reference["host_matrix"][case].values()), "preexisting policy cohorts must stay explicit"

    @pytest.mark.parametrize("case,host", COMPONENT_CASES)
    def test_inactive_original_component_hosts_are_bitwise_develop(comparison, case, host):
        reference, candidate = comparison
        key = f"{case}/{host}"
        old, new = deepcopy(reference["components"][key]), deepcopy(candidate["components"][key])
        # Rebinding sources creates explicit new provenance. The independent
        # task-contract test below verifies unchanged conditions/gates and the
        # exact old revision ancestry; all actual trained state stays strict.
        for packet in (old, new):
            evaluation = packet["applied"]["field_ownership"]["task_contract"]["evaluation"]["value"]
            evaluation.pop("sources", None)
            evaluation.pop("evaluator_revision", None)
        _assert_equal(old, new, key)

    @pytest.mark.parametrize("case", CASES)
    def test_inactive_word_joint_and_resume_are_bitwise_develop(comparison, case):
        reference, candidate = comparison
        old, new = reference["words"][case], candidate["words"][case]
        _assert_equal(old, {name: value for name, value in new.items() if name != "restored"}, case)
        for name in ("suffix", "final"):
            _assert_equal(old[name], new["restored"][name], case + ".resume." + name)

    def test_integrated_canonical_contracts_preserve_original_task_conditions_and_gates(comparison):
        reference, candidate = comparison
        assert reference["declared_tasks"].keys() == candidate["declared_tasks"].keys()
        for name, original in reference["declared_tasks"].items():
            updated = candidate["declared_tasks"][name]
            _assert_equal(original["id"], updated["id"], name + ".id")
            _assert_equal(original["adapter"], updated["adapter"], name + ".adapter")
            execution = dict(updated["execution"])
            execution.pop("transport_consumer", None)
            execution.pop("transport_contract", None)
            _assert_equal(original["execution"], execution, name + ".execution")
            old_eval = {key: value for key, value in original["evaluation"].items()
                        if key not in ("sources", "evaluator_revision")}
            new_eval = {key: value for key, value in updated["evaluation"].items()
                        if key not in ("sources", "evaluator_revision")}
            _assert_equal(old_eval, new_eval, name + ".evaluation")
            old_revision = original["evaluation"].get("evaluator_revision")
            revision = updated["evaluation"].get("evaluator_revision")
            if revision != old_revision:
                assert revision["original_task_commit"] == DEVELOP
                assert revision["original_task_path"] == f"configs/forge/tasks/{name}.json"
                _assert_equal(old_revision, revision["previous_revision"], name + ".original-revision")
            for path, old_hash in original["evaluation"].get("sources", {}).items():
                new_hash = updated["evaluation"]["sources"][path]
                if new_hash != old_hash:
                    assert revision["previous_source_sha256"][path] == old_hash
