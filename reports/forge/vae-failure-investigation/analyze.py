"""Zero-update, source-bound diagnosis of the archived VAE failure.

Run this file with --source-root pointing to the frozen bee9f06ed checkout.
The archived inputs are required; missing evidence never triggers training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def state_hash(state):
    """Exact recursive digest, including tensor bytes and RNG/optimizer state."""
    import torch
    digest = hashlib.sha256()

    def visit(value):
        digest.update(type(value).__name__.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
    visit(state)
    return digest.hexdigest()


def validate_inputs(protocol, evidence_root, source_root):
    """Verify receipts, checkpoints and original scientific bytes before load."""
    records = {}
    for mode, pin in protocol["inputs"].items():
        directory = Path(evidence_root) / f"model_vae_{mode}" / "api"
        receipt_path, checkpoint = directory / "receipt.json", directory / "final-state.pt"
        for path, expected in ((receipt_path, pin["receipt_sha256"]),
                               (checkpoint, pin["checkpoint_sha256"])):
            if file_hash(path) != expected:
                raise ValueError(f"archived evidence hash mismatch: {path}")
        receipt = json.loads(receipt_path.read_text())
        if (receipt["source"]["commit"] != protocol["execution_commit"]
                or not receipt["source_unchanged"] or receipt["completed_updates"] != 1200
                or not receipt["default_protocol_complete"]
                or receipt["case"]["id"] != f"api-model-vae-{mode}"
                or stable_hash(receipt["recipe"]) != pin["recipe_sha256"]
                or receipt["artifacts"]["final-state.pt"]["sha256"] != pin["checkpoint_sha256"]
                or receipt["verdict"] != pin["archived_verdict"]):
            raise ValueError("archived VAE cohort contract mismatch")
        # All package sources are checked, including optimizer/checkpoint code.
        # The frozen provider and constructor-RNG helper complete this path.
        sources = {path: expected for path, expected in receipt["source"]["files_sha256"].items()
                   if path.startswith("particlegan/") or path in (
                       "benchmarks/__init__.py", "benchmarks/toy_audit/__init__.py",
                       "benchmarks/toy_audit/api_model_vae.py",
                       "benchmarks/toy_audit/reproducibility.py")}
        if not sources or stable_hash(sources) != pin["scientific_sources_sha256"]:
            raise ValueError("incomplete archived scientific source binding")
        for path, expected in sources.items():
            if file_hash(Path(source_root) / path) != expected:
                raise ValueError(f"frozen scientific source differs: {path}")
        records[mode] = (receipt, checkpoint)
    return records


@contextmanager
def deterministic_cpu():
    import torch
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def attribute_rows(posterior_mass, rows, decoder_means, likelihood, centers, *, low_usage=.01):
    """Attribute actual prior-sample errors to expected posterior row mass."""
    import torch
    anchors = decoder_means.new_tensor([[-1.5, 0.], [1.5, 0.]])
    distances, modes = torch.cdist(decoder_means, anchors).min(1)
    likelihood_distances, _ = torch.cdist(likelihood, anchors).min(1)
    center_distances, center_modes = torch.cdist(centers, anchors).min(1)
    low = posterior_mass < low_usage
    bad, likelihood_bad = distances > .2, likelihood_distances > .2
    records = []
    for row in range(len(posterior_mass)):
        selected = rows == row
        records.append(dict(row=row, posterior_mass=float(posterior_mass[row]),
                            prior_draws=int(selected.sum()), decoder_center=centers[row].tolist(),
                            center_distance=float(center_distances[row]), center_mode=int(center_modes[row]),
                            decoder_bad_draws=int(bad[selected].sum()),
                            likelihood_bad_draws=int(likelihood_bad[selected].sum()),
                            decoder_modes=torch.bincount(modes[selected], minlength=2).tolist()))

    def group(mask):
        selected = mask[rows]
        count = int(selected.sum())
        return dict(rows=int(mask.sum()), draws=count,
                    decoder_bad_draws=int(bad[selected].sum()),
                    decoder_bad_rate=float(bad[selected].float().mean()) if count else None,
                    likelihood_bad_draws=int(likelihood_bad[selected].sum()),
                    mean_center_distance=float(center_distances[mask].mean()) if bool(mask.any()) else None)

    low_stats, high_stats = group(low), group(~low)
    bad_count = int(bad.sum())
    share = low_stats["decoder_bad_draws"] / bad_count if bad_count else None
    supported = bool(bad_count and share >= .75 and low_stats["decoder_bad_rate"] is not None
                     and high_stats["decoder_bad_rate"] is not None
                     and low_stats["decoder_bad_rate"] > high_stats["decoder_bad_rate"])
    return dict(rows=records, low_usage=low_stats, high_usage=high_stats,
                effectively_unvisited=group(posterior_mass < 1e-6),
                decoder_bad_draws=bad_count, likelihood_bad_draws=int(likelihood_bad.sum()),
                low_usage_share_of_decoder_errors=share,
                posterior_prior_tv=float((posterior_mass - 1 / len(posterior_mass)).abs().sum() / 2),
                association_prediction_supported=supported)


def gradient_panel(fixture, data, *, posterior_seed, prior_seed, prior_noise_seed):
    """Weighted objective gradients on one held-out batch; never calls backward/step."""
    import torch
    from benchmarks.toy_audit.api_model_vae import OBSERVATION_SIGMA
    streams = {name: torch.Generator().manual_seed(seed) for name, seed in (
        ("posterior", posterior_seed), ("prior", prior_seed), ("prior_noise", prior_noise_seed))}
    latent, _ = fixture.prior.sample(len(data), generator=streams["prior"],
                                    noise_generator=streams["prior_noise"])
    encoded = fixture.recipe.encode(fixture.E(data), fixture.prior, draws=2,
                                    generator=streams["posterior"])
    reconstruction = fixture.G(encoded.codes.flatten(0, 1)).reshape(len(data), 2, 2)
    terms = dict(
        adversarial=fixture.loss.g_loss(fixture.D(fixture.G(latent)), fixture.D(data).detach()),
        variational=fixture.recipe.reconstruction_weight * OBSERVATION_SIGMA**2
                    * encoded.negative_elbo(reconstruction, data, observation_sigma=OBSERVATION_SIGMA),
        prior_regularization=fixture.recipe.prior_regularization(
            fixture.prior.z, regularizer=fixture.prior_regularizer, prior=fixture.prior))
    by_role = dict(encoder=list(fixture.E.parameters()), decoder=list(fixture.G.parameters()),
                   prior=[fixture.prior.z])
    parameters = [parameter for group in by_role.values() for parameter in group]
    vectors, result = {}, {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, parameters, allow_unused=True, retain_graph=True)
        roles, vectors[name], offset = {}, {}, 0
        for role, group in by_role.items():
            chunk = grads[offset:offset + len(group)]
            offset += len(group)
            vector = torch.cat([(torch.zeros_like(p) if grad is None else grad).flatten()
                                for p, grad in zip(group, chunk)]).detach()
            vectors[name][role] = vector
            roles[role] = float(vector.double().norm())
        prior_grad = grads[-1]
        row_norms = (torch.zeros_like(fixture.prior.z) if prior_grad is None else prior_grad).norm(dim=1)
        result[name] = dict(weighted_objective_value=float(term.detach()), gradient_norms=roles,
                            prior_row_gradient_norms=row_norms.tolist(),
                            prior_rows_with_nonzero_gradient=int((row_norms > 0).sum()))
    cosines = {}
    names = list(terms)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            row = {}
            for role in by_role:
                a, b = vectors[first][role].double(), vectors[second][role].double()
                denominator = float(a.norm() * b.norm())
                row[role] = float(a @ b) / denominator if denominator else None
            cosines[f"{first}:{second}"] = row
    return dict(batch_size=len(data), terms=result, cosine=cosines)


def analyze_fixture(fixture, panel):
    """Reproduce one observation, then add row attribution and gradient diagnostics."""
    import torch
    from benchmarks.toy_audit.api_model_vae import (
        OBSERVATION_SIGMA, sample_target, score_outputs, observed_negative_elbo)
    before = state_hash(fixture.state_dict())
    global_rng = torch.get_rng_state().clone()
    parameters = [p for model in (fixture.E, fixture.G, fixture.D, fixture.prior) for p in model.parameters()]
    gradients_before = [None if p.grad is None else p.grad.clone() for p in parameters]
    with torch.no_grad():
        rng = torch.Generator().manual_seed(panel["evaluation_seed"])
        data = sample_target(panel["evaluation_inputs"], rng, "cpu")
        encoded = fixture.recipe.encode(fixture.E(data), fixture.prior, draws=8, generator=rng)
        reconstruction = fixture.G(encoded.codes.flatten(0, 1)).reshape(len(data), 8, 2)
        shuffled = reconstruction[torch.randperm(len(data), generator=rng)]
        latent, rows = fixture.prior.sample(len(data), generator=rng, noise_generator=rng)
        decoder_means = fixture.G(latent)
        likelihood = decoder_means + OBSERVATION_SIGMA * torch.randn(len(data), 2, generator=rng)
        metrics, failed = score_outputs(
            data=data, reconstruction=reconstruction, shuffled=shuffled, posterior_codes=encoded.codes,
            posterior_centers=fixture.prior.means()[encoded.indices], log_probs=encoded.log_probs,
            kl=encoded.kl, negative_elbo=observed_negative_elbo(encoded, reconstruction, data),
            decoder_means=decoder_means, likelihood=likelihood)
        attribution = attribute_rows(encoded.log_probs.exp().mean(0), rows, decoder_means,
                                     likelihood, fixture.G(fixture.prior.means()), low_usage=panel["low_usage_mass"])
    gradients = gradient_panel(fixture, data[:panel["gradient_inputs"]],
                               posterior_seed=panel["posterior_seed"], prior_seed=panel["prior_seed"],
                               prior_noise_seed=panel["prior_noise_seed"])
    after = state_hash(fixture.state_dict())
    if before != after or not torch.equal(global_rng, torch.get_rng_state()):
        raise RuntimeError("analysis mutated checkpoint state or global RNG")
    for parameter, old in zip(parameters, gradients_before):
        if (old is None) != (parameter.grad is None) or (old is not None and not torch.equal(old, parameter.grad)):
            raise RuntimeError("analysis mutated parameter gradient buffers")
    return dict(reproduced_final_metrics=metrics, final_failed_bounds=failed, row_attribution=attribution,
                gradient_panel=gradients, state_sha256=before, state_unchanged=True,
                training_rng_unchanged=True, parameter_gradients_unchanged=True, optimizer_updates=0)


def execute(protocol, evidence_root, source_root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    records = validate_inputs(protocol, evidence_root, source_root)
    sys.path.insert(0, str(Path(source_root).resolve()))
    import particlegan
    from benchmarks.toy_audit import api_model_vae
    # Refuse an already-imported implementation from a different checkout.
    for module in (particlegan, api_model_vae):
        if not Path(module.__file__).resolve().is_relative_to(Path(source_root).resolve()):
            raise ValueError("scientific module imported from a different checkout")
    import torch
    start = time.process_time()
    results = {}
    with deterministic_cpu():
        for mode, (receipt, checkpoint) in records.items():
            fixture = api_model_vae.build_case(f"api-model-vae-{mode}",
                                               recipe=particlegan.Recipe(**receipt["recipe"]))
            fixture.load_state_dict(torch.load(checkpoint, weights_only=False, map_location="cpu"))
            result = analyze_fixture(fixture, protocol["panel"])
            expected = receipt["observations"][-1]["metrics"]
            if result["reproduced_final_metrics"] != expected:
                raise ValueError(f"{mode}: final metrics do not exactly reproduce the archived observation")
            result.update(archived_verdict=receipt["verdict"], final_observation_exactly_reproduced=True,
                          input_binding=deepcopy(protocol["inputs"][mode]))
            results[mode] = result
            print(json.dumps(dict(mode=mode, state_unchanged=True,
                                  diagnostic_association=result["row_attribution"]["association_prediction_supported"])), flush=True)
    elapsed = time.process_time() - start
    if elapsed > protocol["cpu_seconds_cap"]:
        raise RuntimeError("saved-state analysis exceeded its CPU budget")
    result = dict(schema="vae_saved_failure_analysis_v1", protocol_sha256=file_hash(HERE / "protocol.json"),
                  execution_commit=protocol["execution_commit"], analysis_source_sha256=file_hash(__file__),
                  optimizer_updates=0, qualification_input=False, reused_original_training_gifs=True,
                  cpu_seconds=elapsed, runtime=dict(torch=torch.__version__, device="cpu", threads=1), results=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    if file_hash(__file__) != protocol["analysis_source_sha256"]:
        raise ValueError("analysis source changed after protocol freeze")
    execute(protocol, args.evidence_root, args.source_root, args.output)


if __name__ == "__main__":
    main()
