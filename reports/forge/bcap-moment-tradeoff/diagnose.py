"""Bounded saved-state diagnosis; never train, sample, or regrade a task.

Run with the original covariance execution snapshot as --source. The fixed
panels enumerate raw rows (words), or raw rows plus symmetric +/-sigma axis
offsets (mode hold). These panels are diagnostics, not the training Gaussian
sampling law. No historical failure-window gradient is reconstructed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from particlegan import Recipe
    from particlegan.moment_tether import MomentTetherRegularizer
    from experiments.forge.state import state_digest
    from experiments.forge.contracts import stable_hash
    from benchmarks.toy_audit.api_images import (
        WordGenerator, WordEncoder, WordJointCritic, WORDS, WORD_CHARS, word_bank,
    )
    from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    witness = json.loads(args.evidence.read_text())
    progress = json.loads((args.archive / "progress.json").read_text())
    assert witness["source_commit"] == progress["source_commit"]
    assert witness["source_digest"] == progress["source_digest"]
    assert args.source.resolve() == (args.archive / "queue/snapshots" / progress["source_digest"]).resolve()
    queue = json.loads((args.archive / "queue/queue/state.json").read_text())
    result = dict(schema_version=1, scope="saved_moment_tradeoff_diagnostic",
        source_commit=progress["source_commit"], source_digest=progress["source_digest"],
        evidence_sha256=sha(args.evidence), protocol_seed=0, analysis_device="cpu",
        analysis_dtype="float32 models; float64 geometry", qualification_input=False,
        optimizer_updates_added=0, sampling_draws_added=0, training_streams_consumed=0,
        constructor_rng_isolated_and_restored=True,
        diagnostic_model_execution=True,
        gradient_scope="Fixed diagnostic panels at saved acquisition/final states, before any critic update. Not original training batches or failure-window gradients.",
        displacement_scope="Hypothetical prior-only zero-momentum DualNorm displacement before direction blend. No optimizer step executed.",
        original_observations={}, word_failure_details=[], states=[], unavailable=[
            "No mode_hold checkpoint or scored sample tensors at update 1050.",
            "No word model/prior checkpoint at updates 5167 or 6167; scored outputs survive.",
            "Historical adversarial/regularizer gradients and actual individual-row displacements are unavailable at failure windows.",
        ])
    result["analysis_source_files"] = {name: sha(args.source / name) for name in (
        "particlegan/moment_tether.py", "particlegan/gan_loss.py", "particlegan/optim/dualnorm.py",
        "particlegan/optim/direction_blend.py", "benchmarks/toy_audit/api_images.py", "lib/toy_models.py")}

    def checked(path, expected=None):
        if expected is not None:
            assert sha(path) == expected, str(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        return saved, dict(path=str(path), sha256=sha(path), state_sha256=state_digest(saved))

    def stats(values):
        values = values.detach().double()
        return dict(min=float(values.min()), median=float(values.median()), max=float(values.max()))

    def summary_gradient(adv, weighted, lr, smoothing):
        advnorm, regnorm = adv.norm(dim=1), weighted.norm(dim=1)
        dot = (adv * weighted).sum(1)
        cosine = dot / (advnorm * regnorm).clamp_min(1e-30)
        total = adv + weighted
        def move(grad):
            return -lr * grad / (grad.norm(dim=1, keepdim=True).square() + smoothing ** 2).sqrt()
        before, after = move(adv), move(total)
        return dict(adversarial_gradient_norm=float(adv.norm()), weighted_tether_gradient_norm=float(weighted.norm()),
            total_norm_ratio=float(weighted.norm() / adv.norm().clamp_min(1e-30)),
            per_row_norm_ratio=stats(regnorm / advnorm.clamp_min(1e-30)),
            per_row_cosine=stats(cosine), opposed_rows=int((dot < 0).sum()), rows=len(adv),
            adversarial_only_step_norm=stats(before.norm(dim=1)), combined_step_norm=stats(after.norm(dim=1)),
            relative_step_change=float((after - before).norm() / before.norm().clamp_min(1e-30)),
            combined_direction_uphill_for_adversarial_rows=int(((adv * after).sum(1) > 0).sum()))

    def analyze(role, task, saved, certificate, reference, kind):
        saved_before = state_digest(saved)
        state = saved["fixture"]["api_state"] if "fixture" in saved else saved["trainer"]
        z = state["models"]["prior"]["z"].detach().clone().requires_grad_(True)
        recipe = Recipe(**state["recipe"])
        ref = SimpleNamespace(moment_tether_mean=reference["moment_tether_mean"],
            moment_tether_covariance=reference["moment_tether_covariance"],
            moment_tether_count=reference["moment_tether_count"])
        regularizer = MomentTetherRegularizer(ref, eps=recipe.prior_reg_eps)
        with torch.random.fork_rng(devices=[]):
            if task == "mode_hold":
                G = SimpleMLPGenerator(z_dim=4, hidden_dim=96, n_hidden=3)
                D = SimpleMLPDiscriminator(hidden_dim=96, n_hidden=3, fourier=3)
                E = None
            else:
                G, D, E = WordGenerator(), WordJointCritic(), WordEncoder()
        for name, model in (("G", G), ("D", D), ("E", E)):
            if model is not None:
                key = name if E is None else {"G": "generator", "D": "critic", "E": "encoder"}[name]
                model.load_state_dict(state["models"][key], strict=True)
                model.eval().requires_grad_(False)
        modules = [model for model in (G, D, E) if model is not None]
        model_before = [state_digest(model.state_dict()) for model in modules]
        loss = recipe.make_loss()
        if E is None:
            sigma = float(state["models"]["prior"]["sigma"])
            offsets = torch.cat((torch.zeros(1, z.shape[1]), sigma * torch.eye(z.shape[1]),
                                 -sigma * torch.eye(z.shape[1])))
            latent = (z[:, None, :] + offsets[None, :, :]).reshape(-1, z.shape[1])
            adversarial = loss.g_loss(D(G(latent)))
            panel = dict(kind="symmetric_axis_sigma_panel", points=len(latent), sigma=sigma,
                         gaussian_sampling_law=False, rows="all 12 raw locations, each weighted equally")
            path_gradients = None
        else:
            latent = z
            fake = G(latent)
            words = word_bank()
            real_scores = D(torch.cat((words.flatten(1), E(words)), 1))
            adversarial = loss.joint_g_loss(D(torch.cat((fake.flatten(1), latent), 1)), real_scores)
            via_g = loss.joint_g_loss(D(torch.cat((fake.flatten(1), latent.detach()), 1)), real_scores)
            direct = loss.joint_g_loss(D(torch.cat((fake.detach().flatten(1), latent), 1)), real_scores)
            g_g = torch.autograd.grad(via_g, z, retain_graph=True)[0]
            g_direct = torch.autograd.grad(direct, z, retain_graph=True)[0]
            panel = dict(kind="exact_five_row_enumeration", points=5, sampling=False,
                         rows="all five raw locations, each weighted equally")
            path_gradients = dict(through_generator_norm=float(g_g.norm()),
                                 direct_joint_latent_norm=float(g_direct.norm()),
                                 path_cosine=float((g_g * g_direct).sum() / (g_g.norm() * g_direct.norm()).clamp_min(1e-30)))
        adv = torch.autograd.grad(adversarial, z, retain_graph=True)[0]
        unweighted = regularizer(z)
        initial_mean, initial_cov = ref.moment_tether_mean, ref.moment_tether_covariance
        scale = initial_cov.diagonal().clamp_min(0) + recipe.prior_reg_eps
        mean = z.mean(0)
        centered = z - mean
        cov = centered.T @ centered / (len(z) - 1)
        mean_loss = ((mean - initial_mean).square() / scale).mean()
        cov_loss = ((cov - initial_cov).square() / torch.outer(scale, scale)).mean()
        assert torch.allclose(mean_loss + cov_loss, unweighted)
        g_mean = torch.autograd.grad(.01 * mean_loss, z, retain_graph=True)[0]
        g_cov = torch.autograd.grad(.01 * cov_loss, z, retain_graph=True)[0]
        weighted = torch.autograd.grad(.01 * unweighted, z)[0]
        if path_gradients is not None:
            assert torch.allclose(adv, g_g + g_direct, atol=1e-6, rtol=1e-5)
            path_gradients["sum_matches_total"] = True
        optimizers = state["optimizers"]
        if isinstance(optimizers, dict):
            optimizers = list(optimizers.values())
        record = dict(role=role, task=task, kind=kind, step=state["completed_steps"], checkpoint=certificate,
            actual_prior_regularizer=recipe.prior_regularizer, actual_weight=recipe.prior_reg,
            diagnosed_weight=.01, baseline_tether_scope=("counterfactual at saved baseline state" if role == "control" else "actual configured tether"),
            diagnostic_target="none; no qualification gate", panel=panel,
            mean_discrepancy=float(mean_loss.detach()), covariance_discrepancy=float(cov_loss.detach()),
            weighted_mean_gradient_norm=float(g_mean.norm()), weighted_covariance_gradient_norm=float(g_cov.norm()),
            mean_cov_gradient_cosine=float((g_mean * g_cov).sum() / (g_mean.norm() * g_cov.norm()).clamp_min(1e-30)),
            prior_shape=list(z.shape), initial_mean=initial_mean.tolist(), initial_covariance=initial_cov.tolist(),
            independent_moment_constraints=z.shape[1] + z.shape[1] * (z.shape[1] + 1) // 2,
            prior_scalar_parameters=z.numel(),
            gradient=summary_gradient(adv, weighted, recipe.lr * recipe.prior_lr_mult, recipe.optimizer_smoothing),
            cumulative_direction_blend=[o["direction_blend"]["stats"] for o in optimizers if "direction_blend" in o],
            joint_latent_gradient_paths=path_gradients)
        if E is not None:
            with torch.no_grad():
                encoded = E(words)
                generated = G(z)
                labels = ["".join(WORD_CHARS[int(c)] for c in row) for row in generated.argmax(1)]
                distances = torch.cdist(encoded, z)
                targets = words.argmax(1)
                reconstructed = G(encoded)
                correct_probs = reconstructed.gather(1, targets[:, None, :]).squeeze(1)
            record["word_alignment"] = dict(prior_row_generated_words=labels,
                encoded_to_prior_distance=distances.tolist(), closest_prior_row=distances.argmin(1).tolist(),
                per_word_nearest_prior_distance=distances.min(1).values.tolist(),
                per_word_reconstruction_min_probability=correct_probs.min(1).values.tolist(),
                word_order=list(WORDS))
        assert model_before == [state_digest(model.state_dict()) for model in modules]
        assert saved_before == state_digest(saved)
        record["diagnostic_model_state_unchanged"] = True
        record["loaded_checkpoint_state_unchanged"] = True
        result["states"].append(record)
        print(json.dumps(dict(event="diagnosed_saved_state", role=role, task=task, kind=kind,
                             step=record["step"], force_ratio=record["gradient"]["total_norm_ratio"])), flush=True)

    for task in ("mode_hold", "five_word_joint_hold"):
        candidate_witness = witness["witnesses"]["moment_tether_0.01"][task]
        csaved, _ = checked(Path(candidate_witness["artifact_root"]) / "state.pt")
        cstate = csaved["fixture"]["api_state"] if "fixture" in csaved else csaved["trainer"]
        reference = cstate["models"]["prior"]
        for role, tasks in witness["witnesses"].items():
            w = tasks[task]
            root = Path(w["artifact_root"])
            receipt_path = root / "adapter-receipt.json"
            assert sha(receipt_path) == w["original_files_sha256"]["adapter-receipt.json"]
            receipt = json.loads(receipt_path.read_text())
            candidate_receipt = json.loads((Path(candidate_witness["artifact_root"]) / "adapter-receipt.json").read_text())
            assert receipt["initialization"] == candidate_receipt["initialization"]
            assert receipt["initial_prior"] == candidate_receipt["initial_prior"]
            observations = receipt["evidence"]["observations"]
            if task == "mode_hold":
                selected = [row for row in observations if row["step"] in (1050, 1200)]
                terminal = observations[-5:]
                window = dict(observations=5, first_step=terminal[0]["step"], last_step=terminal[-1]["step"],
                    minimum_hq=dict(score=min(row["hq"] for row in terminal), target=">= 0.9"),
                    minimum_modes=dict(score=min(row["modes"] for row in terminal), target=">= 8"))
            else:
                selected = [observations[-1]]
                window = None
            result["original_observations"][f"{role}:{task}"] = dict(
                artifact_root=str(root), receipt_sha256=sha(receipt_path),
                recorded_gate=w["gate_status"], original_observation_count=len(observations),
                original_observations_sha256=stable_hash(observations),
                selected_original_observations=selected, terminal_five_window=window)
            checkpoint = w["checkpoint"]
            saved, cert = checked(Path(checkpoint["artifact_root"]) / checkpoint["path"], checkpoint["sha256"])
            assert cert["state_sha256"] == checkpoint["state_sha256"]
            analyze(role, task, saved, cert, reference, "final")
            if task == "five_word_joint_hold":
                descriptor = receipt["evidence"]["saved_observer_outputs"]
                records, records_cert = checked(root / descriptor["path"], descriptor["sha256"])
                for record in records:
                    if record["passed"]:
                        continue
                    for cohort, views, metrics in (
                            ("primary", record["views"], record["metrics"]),
                            ("confirmation", record["confirmation_views"], record["confirmation"]["metrics"])):
                        target = views[0]["target"].argmax(1)
                        probabilities = views[0]["samples"].gather(1, target[:, None, :]).squeeze(1)
                        flat = int(probabilities.argmin())
                        word, pos = divmod(flat, probabilities.shape[1])
                        generated = next(v for v in views if v["title"] == "Actual generated word strings")
                        result["word_failure_details"].append(dict(role=role, step=record["step"], cohort=cohort,
                            saved_outputs=records_cert, original_metrics=metrics, generated_counts=dict(Counter(generated["sample_labels"])),
                            reconstructed_words=views[0]["sample_labels"], lowest_target_token=dict(
                                word=views[0]["target_labels"][word], position_zero_based=pos,
                                token=WORD_CHARS[int(target[word, pos])], probability=float(probabilities[word, pos])),
                            qualification_regraded=False))
                parent = queue["jobs"][w["own_continuation"]["parent_compatibility_key"]]
                attempt = next(a for a in parent["attempts"] if a["attempt_id"] == parent["result"]["attempt_id"])
                parent_root = Path(attempt["path"])
                parent_receipt = json.loads((parent_root / "adapter-receipt.json").read_text())
                parent_descriptor = parent_receipt["evidence"]["checkpoint"]
                acquired_path = Path(parent_receipt["evidence"]["artifact_root"]) / parent_descriptor["path"]
                acquired, acertificate = checked(acquired_path, parent_descriptor["sha256"])
                assert acertificate["state_sha256"] == parent_descriptor["state_sha256"]
                assert acertificate["sha256"] == w["own_continuation"]["parent_checkpoint_sha256"]
                analyze(role, task, acquired, acertificate, reference, "own_acquisition")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="diagnostic_complete", states=len(result["states"]),
                         original_failure_cohorts=len(result["word_failure_details"]), output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
