"""Same-target, fixed-bias recovery diagnostic for a *qualified* cold ring.

The perturbation is a single predeclared +0.35 shift of the generator's x
output bias and its matching EMA entry. D, prior, Adam moments, target data,
rates and random streams are left intact. Both trained arms use the existing
exact post-update resumer. A frozen perturbed generator is evaluated on the
same absolute host noise clock, without gradient or optimizer updates.

The 50-update filter is a local responsiveness diagnostic. A miss at 50 is
not a claim of irrecoverability, and a pass is not a substitute for cold
acquisition, longer same-target hold, or the shared production gate.
"""

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPGenerator
from benchmarks.toy100.continuous_probe import _noise_policy
from benchmarks.transfer_suite.legacy_noise_adapters import wrap_output
from particlegan import ParticlePrior
from reports.toy100.pr84_critic_refinement_capture import _clone, _sha, snapshot
from reports.toy100.pr84_critic_refinement_finite import METHOD, pr84_critic_refinement_finite
from reports.toy100.pr84_critic_refinement_resume import (
    FirstHoldFailure, NOISE_HORIZON, RATES, clean_support, resume_mode_hold,
)


ROOT = Path(__file__).resolve().parents[2]
BIAS = (0.35, 0.0)
RESPONSE_UPDATES = 50
MIN_STABLE_CHECKS = 5  # Existing locked-shared sustained-check count.


def _passes(row):
    return row["modes"] == mode_hold.N_MODES and row["hq"] >= mode_hold.PASS_HQ


@torch.no_grad()
def add_model_error(local):
    """Apply one output-space translation to live G and its matching EMA."""
    wrapped = local["generator"]
    model = getattr(wrapped, "model", wrapped)
    if not isinstance(model, SimpleMLPGenerator) or not isinstance(model.net[-1], torch.nn.Linear):
        raise ValueError("declared SimpleMLPGenerator final affine changed")
    last = model.net[-1]
    if last.out_features != 2 or last.bias is None:
        raise ValueError("declared two-coordinate output bias changed")
    parameters = list(wrapped.parameters())
    matches = [index for index, parameter in enumerate(parameters) if parameter is last.bias]
    if len(matches) != 1 or len(local["ema_g"]) != len(parameters):
        raise ValueError("generator-to-EMA parameter alignment changed")
    index = matches[0]
    key = next(name for name, parameter in wrapped.named_parameters() if parameter is last.bias)
    before = snapshot(local)
    clean_before = clean_support(local)
    delta = last.bias.new_tensor(BIAS)
    last.bias.add_(delta)
    local["ema_g"][index].add_(delta)
    clean_after = clean_support(local)
    after = snapshot(local)
    expected = deepcopy(after)
    expected["generator"][key] = before["generator"][key]
    expected["ema_g"][index] = before["ema_g"][index]
    if _sha(expected) != _sha(before):
        raise RuntimeError("model-error injection changed a non-bias model, optimizer, EMA, RNG or clock state")
    functional_shift = clean_after - clean_before
    if not torch.allclose(functional_shift, delta.expand_as(functional_shift), atol=1e-6, rtol=0):
        raise RuntimeError("output-bias perturbation did not create the declared functional shift")
    return dict(bias=BIAS, generator_bias_key=key, ema_parameter_index=index,
                before_state_sha256=_sha(before), after_state_sha256=_sha(after),
                clean_before=clean_before, clean_perturbed=clean_after,
                actual_clean_shift_rms=float(functional_shift.square().sum(-1).mean().sqrt()),
                actual_clean_shift_max=float(functional_shift.norm(dim=-1).max()))


@contextmanager
def resume_model_error(recorder, generated_source, saved, *, completed_steps,
                       target_steps, log=None):
    """Inject once, after exact restore and before the first resumed noise clock."""
    with resume_mode_hold(recorder, generated_source, saved,
                          completed_steps=completed_steps, target_steps=target_steps,
                          fail_fast=False, log=log) as state:
        ordinary_before = state.before_step
        state.model_error = None
        state.perturbed_start_snapshot = None

        def before_step(step, local):
            ordinary_before(step, local)
            if step == completed_steps:
                if state.model_error is not None:
                    raise RuntimeError("model error was injected more than once")
                state.model_error = add_model_error(local)
                state.perturbed_start_snapshot = snapshot(local)
                # Movement accounting now starts at the perturbed function.
                state.before_clock = state.perturbed_start_snapshot
                state._before_support = state.model_error["clean_perturbed"]

        state.before_step = before_step
        yield state


def run_continuation(saved, recipe, noise, *, completed_steps, target_steps,
                     perturb=False, fail_fast=False, log=None):
    """Run one source-bound branch; also reusable for the unperturbed long hold."""
    from benchmarks import learned_lr_evaluation as bridge
    from benchmarks.smart_descent import evaluate
    from benchmarks.transfer_suite import vector_tasks
    from benchmarks.transfer_suite.compare_defaults import candidate, optimizer_defaults

    if perturb and fail_fast:
        raise ValueError("perturbed local response needs all declared checkpoints")
    if saved["noise"]["step_calls"] != completed_steps or target_steps <= completed_steps:
        raise ValueError("expected a complete pre-clock snapshot and positive continuation")
    policy = _noise_policy(noise, NOISE_HORIZON)
    applied = []
    result = None
    with ExitStack() as stack:
        stack.enter_context(optimizer_defaults(recipe, applied))
        control = evaluate.FixedControl(vector_tasks.fixed_policy("cosine"), NOISE_HORIZON)
        stack.enter_context(bridge.control_host_schedules(control))
        recorder, generated = stack.enter_context(pr84_critic_refinement_finite())
        if perturb:
            state = stack.enter_context(resume_model_error(recorder, generated, saved,
                completed_steps=completed_steps, target_steps=target_steps, log=log))
        else:
            state = stack.enter_context(resume_mode_hold(recorder, generated, saved,
                completed_steps=completed_steps, target_steps=target_steps,
                fail_fast=fail_fast, log=log))
        common = candidate(recipe)
        try:
            result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(
                steps=target_steps, particle_l2=0., vicreg_weight=recipe.prior_reg),
                gan_factory=common.make_loss, cap_factory=common.make_penalty,
                noise_policy=policy, diagnostics=True)
        except FirstHoldFailure:
            if not fail_fast or state.failure is None:
                raise
    receipt = state.receipt()
    if (not receipt["restored_before_set_step"] or
            receipt["actual_adam_updates"] != {"d": receipt["updates"], "g": receipt["updates"]} or
            receipt["optimizer_callbacks"] != {"d": 3 * receipt["updates"],
                                              "g": 3 * receipt["updates"]} or
            state.final_state["noise_policy"]["_step_calls"] != completed_steps + receipt["updates"] or
            tuple(group["lr"] for group in recorder.optimizers[0].param_groups) != RATES["d"] or
            tuple(group["lr"] for group in recorder.optimizers[1].param_groups) != RATES["g"]):
        raise RuntimeError("continued rate, moment or original noise-clock accounting changed")
    return dict(result=result, receipt=receipt, dynamics=recorder.receipt(),
                state=state.final_state, applied=applied, policy=policy.receipt(),
                model_error=getattr(state, "model_error", None),
                perturbed_start=getattr(state, "perturbed_start_snapshot", None),
                final_clean_support=clean_support(recorder._local),
                host_source_sha256=recorder.host_source["generated_function_sha256"])


@torch.no_grad()
def frozen_perturbed_points(saved, noise, *, completed_steps, target_steps):
    """No-update quality control; host evaluation draws use the same step IDs."""
    policy = _noise_policy(noise, NOISE_HORIZON)
    if policy.output_scale is not None or saved["noise_policy"]["output_scale"] is not None:
        raise ValueError("frozen control requires the fixed output-noise scale")
    keys = set(vars(policy)) - {"input_stream", "output_stream"}
    if keys != set(saved["noise_policy"]) or saved["noise"]["step_calls"] != completed_steps:
        raise ValueError("frozen control state/clock schema changed")
    model = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2)
    generator = wrap_output(model, policy)
    if policy.generator_base_parameters != saved["noise_policy"]["generator_base_parameters"]:
        raise ValueError("frozen control generator size changed")
    vars(policy).update(_clone(saved["noise_policy"]))
    policy.input_stream.set_state(saved["rng"]["input"])
    if (policy.output_stream is None) != (saved["rng"]["output"] is None):
        raise ValueError("frozen output RNG type changed")
    if policy.output_stream is not None:
        policy.output_stream.set_state(saved["rng"]["output"])
    generator.load_state_dict(saved["generator"])
    prior = ParticlePrior(*saved["prior"]["z"].shape,
                          generator=torch.Generator().manual_seed(0))
    prior.load_state_dict(saved["prior"])
    means = mode_hold.ring_means()
    initial_parameters = _sha(dict(generator=generator.state_dict(), prior=prior.state_dict()))
    points = []
    with torch.random.fork_rng(devices=[]):
        for step in range(completed_steps + 1, target_steps + 1):
            policy.set_step(step - 1)
            with policy.evaluation(step):
                latent, _ = prior.sample(mode_hold.EVAL_N,
                    generator=torch.Generator().manual_seed(9))
                point = mode_hold.diversity(generator(latent), means)
                support = mode_hold.diversity(generator(prior.z), means, detailed=True)
            point.update(step=step, support=support, support_scope="one noisy draw per particle",
                         output_sigma=policy.output_sigma)
            points.append(point)
    if (_sha(dict(generator=generator.state_dict(), prior=prior.state_dict())) != initial_parameters
            or policy._step_calls != target_steps
            or len(policy._effective_step_trace) != target_steps):
        raise RuntimeError("frozen control moved weights or changed the original noise clock")
    return dict(points=points, noise_step_calls=policy._step_calls,
                output_sigma=policy.output_sigma, input_sigma=policy.input_sigma,
                effective_step_trace=policy._effective_step_trace[completed_steps:],
                fixed_model_state_sha256=initial_parameters)


def grade_response(control, perturbed, frozen):
    """Use the existing ring thresholds and five-check suffix only."""
    base, changed, fixed = (value["receipt"]["checkpoints"] if "receipt" in value
                            else value["points"] for value in (control, perturbed, frozen))
    if not (len(base) == len(changed) == len(fixed) == RESPONSE_UPDATES):
        raise ValueError("incomplete predeclared response window")
    expected = list(range(base[0]["step"], base[0]["step"] + RESPONSE_UPDATES))
    if any([row["step"] for row in branch] != expected for branch in (base, changed, fixed)):
        raise ValueError("response arms used different absolute checkpoint clocks")
    control_all_pass = all(_passes(row) for row in base)
    frozen_all_fail = all(not _passes(row) for row in fixed)
    response_suffix_pass = all(_passes(row) for row in changed[-MIN_STABLE_CHECKS:])
    return dict(status="PASS_LOCAL_RESPONSE" if (control_all_pass and frozen_all_fail
                and response_suffix_pass) else "FAIL_LOCAL_RESPONSE_FILTER",
                steps=RESPONSE_UPDATES, first=expected[0], last=expected[-1],
                control_all_pass=control_all_pass, frozen_all_fail=frozen_all_fail,
                perturbed_final_five_pass=response_suffix_pass,
                perturbation=BIAS, original_thresholds=dict(modes=8, hq=.9),
                no_signal_movement_floor=False,
                interpretation="bounded same-target responsiveness only; a 50-update miss is not irrecoverability")


def require_qualified_cold(directory):
    """Read all source and gate bytes before making a response artifact."""
    declaration = json.loads((directory / "declaration.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    ring = json.loads((directory / "mode_hold.json").read_text())
    trajectory = json.loads((directory / "trajectory.json").read_text())
    if (declaration["method"] != METHOD or summary["method"] != METHOD
            or summary["status"] != "PASS" or ring["verdict"]["status"] != "PASS"
            or trajectory["verdict"]["status"] != "PASS"
            or ring["source"] != declaration["source"]
            or trajectory["source"] != declaration["source"]
            or [row["task"] for row in summary["stages"]] != ["trajectory", "mode_hold"]
            or not all(row["verdict"]["passed"] for row in summary["stages"])
            or ring["dynamics"]["outer_steps"] != 1200
            or ring["dynamics"]["method"] != METHOD
            or ring["noise"]["step_calls"] != 1200
            or any(ring["actual_adam_accounting"][role]["actual_adam_calls"] != 1200
                   for role in ("d", "g"))):
        raise RuntimeError("complete repaired cold acquisition did not pass")
    for name, digest in declaration["source"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"cold acquisition source changed: {name}")
    path = directory / ring["final_state_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != ring["final_state_file_sha256"]:
        raise RuntimeError("cold ring snapshot bytes changed")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if (saved["noise"]["step_calls"] != 1200
            or saved["noise_policy"]["total_steps"] != NOISE_HORIZON):
        raise RuntimeError("cold ring snapshot is not the completed original-budget boundary")
    return saved, declaration, summary, ring


def main():
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved, cold_declaration, _, ring = require_qualified_cold(args.cold)
    config = json.loads((args.cold / "config.json").read_text())
    recipe, noise, _ = declared_recipe(config)
    torch.set_num_threads(1)
    completed, target = 1200, 1200 + RESPONSE_UPDATES
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in sorted(set(cold_declaration["source"]) | {
                  "reports/toy100/pr84_critic_refinement_resume.py",
                  "reports/toy100/pr84_model_error_recovery.py",
                  "reports/toy100/pr84_critic_refinement_capture.py",
                  "benchmarks/locked_shared/mlp.py",
                  "benchmarks/transfer_suite/legacy_noise_adapters.py"})}
    args.output.mkdir(parents=True, exist_ok=False)
    for name in source:
        target_file = args.output / "source" / name
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes((ROOT / name).read_bytes())
    declaration = dict(method="fixed_g_output_bias_same_target_response",
        source=source, cold_ring_state_sha256=ring["final_state_file_sha256"],
        cold_declaration_sha256=hashlib.sha256((args.cold / "declaration.json").read_bytes()).hexdigest(),
        steps=RESPONSE_UPDATES, bias=BIAS,
        acceptance="unperturbed 50/50 original ring PASS; frozen perturbed 50/50 FAIL; perturbed final5/5 PASS",
        same_target=True, noise_horizon=NOISE_HORIZON, rates=RATES,
        gate_scope="local diagnostic only, not long hold or common production gate")
    (args.output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    print(json.dumps(dict(event="DECLARED", **declaration)), flush=True)
    try:
        control = run_continuation(saved, recipe, noise, completed_steps=completed,
                                   target_steps=target)
        perturbed = run_continuation(saved, recipe, noise, completed_steps=completed,
                                     target_steps=target, perturb=True)
        frozen = frozen_perturbed_points(perturbed["perturbed_start"], noise,
                                         completed_steps=completed, target_steps=target)
        for branch in (control, perturbed):
            if (branch["state"]["noise_policy"]["_effective_step_trace"][completed:]
                    != frozen["effective_step_trace"]):
                raise RuntimeError("paired arms used different absolute output-noise clocks")
        if (_sha(control["state"]["rng"]) != _sha(perturbed["state"]["rng"])
                or control["state"]["noise_policy"]["_counts"] !=
                   perturbed["state"]["noise_policy"]["_counts"]):
            raise RuntimeError("paired trained arms consumed different random draws or noise calls")
        verdict = grade_response(control, perturbed, frozen)
        baseline_clean = perturbed["model_error"]["clean_before"]
        shifted_clean = perturbed["model_error"]["clean_perturbed"]
        final_changed = perturbed["final_clean_support"]
        final_control = control["final_clean_support"]
        def rms(value):
            return float(value.square().sum(-1).mean().sqrt())
        functional = dict(initial_bias_shift_rms=rms(shifted_clean - baseline_clean),
            perturbed_training_move_rms=rms(final_changed - shifted_clean),
            final_indexed_distance_to_original_rms=rms(final_changed - baseline_clean),
            control_training_move_rms=rms(final_control - baseline_clean),
            initial_clean=baseline_clean.tolist(), final_control_clean=final_control.tolist(),
            final_perturbed_clean=final_changed.tolist(),
            interpretation="indexed function movement is descriptive; particle permutation can preserve the data law")
        def slim(branch):
            return dict(result=branch["result"], receipt=branch["receipt"],
                        dynamics=branch["dynamics"], applied=branch["applied"],
                        policy=branch["policy"], final_state_sha256=_sha(branch["state"]),
                        host_source_sha256=branch["host_source_sha256"])
        perturbation = dict(perturbed["model_error"])
        perturbation["clean_before"] = perturbation["clean_before"].tolist()
        perturbation["clean_perturbed"] = perturbation["clean_perturbed"].tolist()
        report = dict(verdict=verdict, unperturbed=slim(control),
                      perturbed=slim(perturbed), frozen=frozen,
                      perturbation=perturbation, functional_displacement=functional,
                      same_target=True,
                      trained_condition="source-bound constant-rate PR84 repaired critic refinement",
                      frozen_scope="no training steps; exact host evaluation law at each absolute noise clock")
        (args.output / "response.json").write_text(json.dumps(report, allow_nan=False) + "\n")
        print(json.dumps(dict(event="RESPONSE_DONE", **verdict)), flush=True)
    except BaseException as error:
        (args.output / "error.json").write_text(json.dumps(dict(status="ERROR_INCOMPLETE",
            error=repr(error), shared_gate_eligible=False), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
