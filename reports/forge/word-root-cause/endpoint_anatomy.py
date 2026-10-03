"""Deterministic CPU anatomy of saved WordFixture states; no optimizer updates.

All five prior rows are evaluated exactly. No latent/data/noise draws are made.
The atom integral is diagnostic and cannot replace a task evaluation receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from benchmarks.toy_audit.api_images import (
    WORDS, WORD_CHARS, WordGenerator, WordEncoder, WordJointCritic, word_bank)
from particlegan import init
from particlegan.particle_prior import ParticlePrior
from particlegan.vicreg_loss import ParticleRegularizer
from experiments.forge.state import state_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(probabilities):
    return ["".join(WORD_CHARS[int(i)] for i in row) for row in probabilities.argmax(1)]


def jacobian_sizes(generator, latents):
    out = []
    for latent in latents:
        jac = torch.autograd.functional.jacobian(
            lambda z: generator(z[None]).flatten(), latent.detach(), vectorize=True)
        logit_jac = torch.autograd.functional.jacobian(
            lambda z: generator.net(z[None]).flatten(), latent.detach(), vectorize=True)
        out.append({"probability_jacobian_frobenius": float(jac.norm()),
                    "probability_jacobian_singular_values": torch.linalg.svdvals(jac).tolist(),
                    "logit_jacobian_frobenius": float(logit_jac.norm()),
                    "zero_probability_derivatives": int((jac == 0).sum()),
                    "probability_derivatives": jac.numel()})
    return out


def analyze(directory):
    state_path = directory / "state.pt"
    fixture = torch.load(state_path, map_location="cpu", weights_only=True)["fixture"]
    state = fixture["api_state"]
    receipt = json.loads((directory / "adapter-receipt.json").read_text())
    with torch.random.fork_rng(devices=[]):
        G, E, D = WordGenerator(), WordEncoder(), WordJointCritic()
        initial_prior = ParticlePrior(num_particles=5, z_dim=2,
            generator=torch.Generator().manual_seed(0))
        init.deterministic_orthogonal_(initial_prior,
            parameter_seeds=receipt["initialization"]["prior"]["parameter_seeds"])
    for role, model in (("generator", G), ("encoder", E), ("critic", D)):
        model.load_state_dict(state["models"][role])
        model.eval().requires_grad_(False)
    words = word_bank()
    prior = state["models"]["prior"]["z"].detach()
    encoded = E(words).detach()
    generated, recon = G(prior).detach(), G(encoded).detach()
    logits = G.net(prior).reshape(5, 28, 6).detach()
    distance = torch.cdist(encoded, prior)
    nearest = distance.argmin(1)
    labels = decode(generated)
    canonical = [w + "_" for w in WORDS]
    confidence = generated.amax(1).amin(1)
    valid = [label in canonical and float(p) >= .9 for label, p in zip(labels, confidence)]
    masses = [sum(v and label == word for v, label in zip(valid, labels)) / 5 for word in canonical]
    rejection = 1 - sum(valid) / 5
    original = initial_prior.z.detach()
    reconstructed_initial_hash = state_digest(initial_prior.state_dict())
    if reconstructed_initial_hash != receipt["evidence"]["host"]["initial_prior_sha256"]:
        raise ValueError("saved named initialization does not reproduce initial prior hash")
    interpolation = []
    for t in (0., .5, 1., 1.5, 2.):
        latent = prior[nearest] + t * (encoded - prior[nearest])
        output = G(latent).detach()
        raw_logits = G.net(latent).reshape(5, 28, 6).detach()
        correct_logits = (raw_logits * words).sum(1)
        rivals = raw_logits.masked_fill(words.bool(), float("-inf")).amax(1)
        interpolation.append({"fraction": t, "decoded": decode(output),
            "minimum_correct_probability": (output * words).sum(1).amin(1).tolist(),
            "minimum_correct_logit_margin": (correct_logits - rivals).amin(1).tolist()})
    real_joint = torch.cat((words.flatten(1), encoded), 1).requires_grad_()
    fake_joint = torch.cat((generated.flatten(1), prior), 1).requires_grad_()
    real_grad = torch.autograd.grad(D(real_joint).sum(), real_joint)[0]
    fake_grad = torch.autograd.grad(D(fake_joint).sum(), fake_joint)[0]
    return {
        "directory": str(directory.resolve()),
        "artifacts": {name: {"sha256": sha(directory / name), "bytes": (directory / name).stat().st_size}
                      for name in ("state.pt", "adapter-receipt.json")},
        "completed_updates": state["completed_steps"], "recipe": fixture["recipe"],
        "prior": {"rows": prior.tolist(), "initial_rows": original.tolist(),
            "initial_sha256": reconstructed_initial_hash,
            "distance_from_initial_by_row": (prior - original).norm(dim=1).tolist(),
            "distance_from_initial_frobenius": float((prior - original).norm()),
            "minimum_pair_distance": float(torch.pdist(prior).min()),
            "maximum_pair_distance": float(torch.pdist(prior).max()),
            "centroid": prior.mean(0).tolist(), "coordinate_std": prior.std(0).tolist(),
            "unweighted_spread_cost": float(ParticleRegularizer()(prior))},
        "generator_atoms": {"decoded": labels, "minimum_argmax_token_probability": confidence.tolist(),
            "canonical_confident": valid,
            "exact_uniform_atom_integral": {"sampling": "enumerate exactly five rows, each weight .2; diagnostic only",
                "accepted_word_masses": masses, "rejected_mass": rejection,
                "quality_fraction": sum(valid) / 5, "modes": sum(m > 0 for m in masses),
                "mass_tv": (sum(abs(m - .2) for m in masses) + rejection) / 2},
            "logit_minimum": float(logits.min()), "logit_maximum": float(logits.max()),
            "maximum_token_logit_range": float((logits.amax(1) - logits.amin(1)).max()),
            "zero_token_probabilities": int((generated == 0).sum()),
            "unit_token_probabilities": int((generated == 1).sum()),
            "jacobians_at_prior": jacobian_sizes(G, prior)},
        "encoder": {"rows": encoded.tolist(), "reconstructed_words": decode(recon),
            "minimum_correct_token_probability_by_word": (recon * words).sum(1).amin(1).tolist(),
            "nearest_prior_indices": nearest.tolist(), "nearest_prior_distances": distance.amin(1).tolist(),
            "interpolation_prior_to_encoder": interpolation,
            "jacobians_at_encoder": jacobian_sizes(G, encoded)},
        "critic": {"real_scores": D(real_joint).detach().tolist(), "fake_scores": D(fake_joint).detach().tolist(),
            "real_word_gradient_norm": real_grad[:, :168].norm(dim=1).tolist(),
            "real_latent_gradient_norm": real_grad[:, 168:].norm(dim=1).tolist(),
            "fake_word_gradient_norm": fake_grad[:, :168].norm(dim=1).tolist(),
            "fake_latent_gradient_norm": fake_grad[:, 168:].norm(dim=1).tolist()},
        "final_optimizer_groups": [[{"lr": g["lr"], "betas": list(g["betas"])}
                                    for g in opt["param_groups"]] for opt in state["optimizers"]],
    }


def support_counterexamples():
    target = word_bank()[:1]
    perturbation = torch.zeros_like(target)
    # Character 'b' at apple's first position has zero target mass.
    perturbation[0, WORD_CHARS.index("b"), 0] = -.029
    negative = target + perturbation
    above_one = target.clone()
    above_one[0, WORD_CHARS.index("a"), 0] += .029
    return {"scope": "fixed arithmetic support counterexamples, no noise draw or estimated frequency",
            "negative_coordinate": float(negative.min()),
            "negative_example_token_mass": float(negative.sum(1)[0, 0]),
            "above_one_coordinate": float(above_one.max()),
            "above_one_example_token_mass": float(above_one.sum(1)[0, 0]),
            "interpretation": "Unprojected additive Gaussian output noise has support outside categorical probability simplices. This establishes a training-law mismatch, not its causal responsibility for any failure."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps({"schema_version": 1, "scope": "saved-state anatomy; no updates, no stochastic samples, no qualification",
        "runtime": {"torch": str(torch.__version__), "device": "cpu", "threads": 1},
        "output_noise_support_counterexamples": support_counterexamples(),
        "states": [analyze(path) for path in args.states]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
