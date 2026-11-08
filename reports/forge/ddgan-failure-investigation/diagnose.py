"""Read original DDGAN checkpoints; perform no training or qualification.

Run in a fresh Python process with --source-root at the original scientific
checkout. Imports of model code happen only after its complete source check.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
FROZEN_COMMIT = "bee9f06ed21695e518c1cb702e12a385ab9389e8"
TASKS = ("model_ddgan_gaussian_chain", "model_ae_ddgan_gaussian_chain")
PANELS = ((-.5, .25), (.5, -.25))
REPORT_ONLY = frozenset(("reports/forge/api-model-toys/run.py",
    "experiments/forge/model_toy_publication.py", "experiments/forge/publication_memory.py"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(source_root, manifest, repair):
    """Accept exact frozen bytes or the already published exact report overlay."""
    if manifest["origin_commit"] != FROZEN_COMMIT:
        raise ValueError("unexpected original scientific source")
    mismatches = []
    for name, expected in manifest["files"].items():
        path = Path(source_root) / name
        actual = sha256(path) if path.is_file() else None
        change = repair.get("reporting_only_changes", {}).get(name, {})
        if actual == expected:
            continue
        if (name in REPORT_ONLY and change.get("frozen_sha256") == expected
                and actual == change.get("repaired_sha256")):
            continue
        mismatches.append(name)
    if mismatches:
        raise ValueError("frozen source mismatch: " + ", ".join(mismatches))
    return dict(origin_commit=FROZEN_COMMIT, manifest_digest=manifest["digest"],
        files_verified=len(manifest["files"]), scientific_source_unchanged=True)


def verify_artifacts(directory, run):
    directory = Path(directory)
    paths = {"receipt.json": (directory / "api/receipt.json", run["raw_receipt_sha256"]),
        "adapter-receipt.json": (directory / "adapter-receipt.json", run["adapter_receipt_sha256"])}
    paths.update({name: (directory / "api" / name, entry["sha256"])
                  for name, entry in run["artifact_hashes"].items()})
    for name, (path, expected) in paths.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("original artifact mismatch: " + name)
    return {name: expected for name, (_, expected) in paths.items()}


def state_digest(models):
    digest = hashlib.sha256()
    def add(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                add(key); add(value[key])
        elif isinstance(value, (list, tuple)):
            for entry in value:
                add(entry)
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")
    for name, model in sorted(models.items()):
        add(name); add(model.state_dict())
        add({key: p.grad for key, p in model.named_parameters()})
        add({key: m.training for key, m in model.named_modules()})
    return digest.hexdigest()


def mechanism(target, schedule, step, xy):
    """Analytic variance budget and Gaussian context rarity, not fitted metrics."""
    xt = target.means.new_tensor([xy])
    _, _, clean_var = target.posterior_parameters(xt, schedule.ab[step])
    learned_var = schedule.A[step].square() * clean_var[0, 0]
    noise_var = schedule.posterior_var[step]
    marginal_var = schedule.ab[step] * target.variances[0] + 1 - schedule.ab[step]
    marginal_mean = schedule.ab[step].sqrt() * target.means[0]
    distance2 = ((xt[0] - marginal_mean).square() / marginal_var).sum()
    return dict(clean_coefficient=float(schedule.A[step]), context_coefficient=float(schedule.B[step]),
        reverse_noise_variance=float(noise_var),
        learned_clean_share_of_exact_transition_variance=(learned_var / (learned_var + noise_var)).tolist(),
        context_mahalanobis=float(distance2.sqrt()),
        gaussian_context_radial_tail=float((-distance2 / 2).exp()))


def critic_alignment(critic, fake, oracle, xt, t, target_mean):
    """Directional derivative for translating all fake samples toward true mean.

    ``descent`` is the negative gradient of each example's non-saturating
    generator loss with respect to its output, before the public batch-mean
    factor 1/n. Positive alignment helps that translation locally. This is not
    a parameter-update or trajectory-level causal claim.
    """
    x = fake.detach().requires_grad_()
    score = critic(x, xt=xt, t=t)
    descent = -torch.autograd.grad(F.softplus(-score).sum(), x)[0]
    correction = target_mean.to(x) - x.detach().mean(0)
    with torch.no_grad():
        oracle_score = critic(oracle, xt=xt, t=t).mean()
    return dict(mean_generator_descent=descent.mean(0).tolist(),
        desired_mean_translation=correction.tolist(),
        mean_correction_descent_alignment=float((descent @ correction).mean()),
        mean_fake_score=float(score.detach().mean()), mean_oracle_score=float(oracle_score))


def panel(api, models, target, streams, step, xy, n, label):
    G, D, prior, schedule = (models[k] for k in ("G", "D", "prior", "schedule"))
    rng = lambda suffix: streams.generator("eval", component=label + suffix, device="cpu")
    xt = torch.tensor(xy).expand(n, 2)
    t = torch.full((n,), step, dtype=torch.long)
    with torch.no_grad():
        z, _ = prior.sample(n, generator=rng("_latent"), noise_generator=rng("_kernel"))
        clean = G(xt, z, t)
        eta = torch.randn(xt.shape, generator=rng("_reverse"))
        fake = schedule.reverse(clean, xt, t, eta)
        weights, means, variances = target.posterior_parameters(xt[:1], schedule.ab[step])
        transition = target.transition_parameters(xy, step, schedule)
        oracle = target.draw(*transition, n, rng("_oracle"))
        clean_metrics = api.distribution_metrics(clean, weights[0], means[0], variances[0])
        transition_metrics = api.distribution_metrics(fake, *transition)
    return dict(context=list(xy), mechanism=mechanism(target, schedule, step, xy),
        clean_posterior=clean_metrics, reverse_transition=transition_metrics,
        critic=critic_alignment(D, fake, oracle, xt, t, (transition[0][:, None] * transition[1]).sum(0)))


def suffix_summary(receipt):
    observations = receipt["observations"]
    suffix = observations[-5:]
    metrics = ("final_projection_cdf", "final_normalized_mean", "worst_transition_cdf",
               "paired_nmse", "paired_worst_axis_nrmse", "paired_shuffled_gain")
    return dict(steps=[o["step"] for o in suffix], passing_joint_observations=sum(o["passed"] for o in suffix),
        ranges={name: dict(min=min(o["metrics"][name] for o in suffix),
                          max=max(o["metrics"][name] for o in suffix))
                for name in metrics if all(name in o["metrics"] for o in suffix)},
        endpoint_metrics=observations[-1]["metrics"], endpoint_failed_bounds=observations[-1]["failed_bounds"])


def analyze(api, named_streams, state, receipt, n=4096, context_count=8):
    """Analyze a detached CPU copy; no optimizer or training RNG is constructed."""
    recipe = state["recipe"]
    if recipe["loss"] != "non_saturating":
        raise ValueError("diagnostic output-gradient formula requires non_saturating loss")
    with torch.random.fork_rng(devices=[]):
        models = dict(G=api.ChainGenerator(recipe["z_dim"]), D=api.ChainCritic(),
            prior=api.MoGParticlePrior(num_particles=recipe["num_particles"], z_dim=recipe["z_dim"],
                sigma=.25, standardize=False), schedule=api.DDGAN(api.ALPHA_BAR))
        # Register the recipe's persistent prior-reference buffers before
        # strict restore; their original values are loaded from the checkpoint.
        api.Recipe(**recipe).apply_prior_policy(models["prior"])
    for name, model in models.items():
        model.load_state_dict(state["models"][name])
        model.eval(); model.requires_grad_(False)
    before, rng_before = state_digest(models), torch.get_rng_state().clone()
    target, streams = api.MixtureTarget("gaussian"), named_streams(0, device="cpu")
    fixed = {f"t{step}_q{q}": panel(api, models, target, streams, step, xy, n, f"t{step}_q{q}")
             for step in range(1, 5) for q, xy in enumerate(PANELS)}
    # Fresh held-out contexts from the actual analytic x_t marginal; diagnostic
    # samples do not replace the original eight frozen qualification panels.
    random_contexts = {}
    for step in range(1, 5):
        rng = streams.generator("eval", component=f"t{step}_contexts", device="cpu")
        contexts = target.sample(context_count, rng) * models["schedule"].ab[step].sqrt()
        contexts += (1 - models["schedule"].ab[step]).sqrt() * torch.randn(contexts.shape, generator=rng)
        values = [panel(api, models, target, streams, step, xy.tolist(), n, f"t{step}_random{i}")
                  for i, xy in enumerate(contexts)]
        random_contexts[f"t{step}"] = dict(contexts=context_count,
            normalized_mean_median=statistics.median(v["reverse_transition"]["normalized_mean"] for v in values),
            normalized_mean_max=max(v["reverse_transition"]["normalized_mean"] for v in values),
            projection_cdf_median=statistics.median(v["reverse_transition"]["projection_cdf"] for v in values),
            projection_cdf_max=max(v["reverse_transition"]["projection_cdf"] for v in values),
            mean_correction_opposed=sum(v["critic"]["mean_correction_descent_alignment"] < 0 for v in values))
    pure = before == state_digest(models) and torch.equal(rng_before, torch.get_rng_state())
    if not pure:
        raise RuntimeError("diagnostic changed loaded models, gradients, modes or caller Torch RNG")
    return dict(completed_training_updates=state["completed_steps"], diagnostic_samples_per_panel=n,
        fixed_panels=fixed, marginal_context_summary=random_contexts, saved_terminal_suffix=suffix_summary(receipt),
        purity=dict(models_gradients_modes_unchanged=True, caller_torch_rng_unchanged=True,
            original_training_streams_consumed=False, optimizer_constructed=False, training_updates_added=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite diagnostic output")
    original = ROOT / "reports/forge/api-model-toys"
    manifest = json.loads((original / "execution-source.json").read_text())
    repair = json.loads((original / "export-repair.json").read_text())
    source = verify_source(args.source_root, manifest, repair)
    # Avoid accidentally analyzing with packages already imported from develop.
    if any(name in sys.modules for name in ("particlegan", "benchmarks", "experiments")):
        raise RuntimeError("run diagnostic in a fresh Python process")
    sys.path.insert(0, str(args.source_root.resolve()))
    api = importlib.import_module("benchmarks.toy_audit.api_model_ddgan")
    named_streams = importlib.import_module("experiments.forge.rng").NamedStreams
    if not Path(api.__file__).resolve().is_relative_to(args.source_root.resolve()):
        raise RuntimeError("scientific module came from a different checkout")
    publication_path = ROOT / "reports/toy_audit/api_contract/model_modes/publication.json"
    publication = json.loads(publication_path.read_text())
    result = dict(schema="ddgan_saved_state_diagnosis_v1", qualification_input=False,
        source=source, original_publication_sha256=sha256(publication_path),
        analysis_code_sha256=sha256(Path(__file__)),
        analysis_scope="exploratory follow-up of previously inspected saved states; no independent validation",
        diagnostic_protocol=dict(seed=0, samples_per_panel=4096, contexts_per_step=8,
            training_updates=0, backend="cpu", threads=1,
            gradient_units="negative per-example non-saturating loss gradient with respect to x_prev, before batch-mean 1/n"), tasks={})
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for task in TASKS:
            print(json.dumps(dict(event="analyze", task=task)), flush=True)
            run = next(r for r in publication["runs"] if r["task_id"] == task)
            directory = args.artifacts / task
            hashes = verify_artifacts(directory, run)
            # The original trusted local checkpoint is loaded only after its
            # committed publication hash has been checked.
            state = torch.load(directory / "api/final-state.pt", map_location="cpu", weights_only=False)
            receipt = json.loads((directory / "api/receipt.json").read_text())
            if (json.loads(json.dumps(state["recipe"])) != run["recipe"] or state["case_id"] != run["id"]
                    or state["seed"] != 0 or state["completed_steps"] != 2500):
                raise ValueError("checkpoint contract differs from published cohort")
            value = analyze(api, named_streams, state, receipt)
            value["original_artifact_hashes"] = hashes
            result["tasks"][task] = value
            # Confirm no external original artifact was changed by analysis.
            verify_artifacts(directory, run)
            print(json.dumps(dict(event="complete", task=task, training_updates_added=0)), flush=True)
    finally:
        torch.set_num_threads(threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(event="written", output=str(args.output), qualification_input=False)), flush=True)


if __name__ == "__main__":
    main()
