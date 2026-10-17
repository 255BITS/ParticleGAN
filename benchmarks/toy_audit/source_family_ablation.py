"""Read-only endpoint mixed-code ablation for saved routed fixture states.

The bank, router weights, encoder and frozen host retain their trained values.
Only mixed codes supplied to generation become zero. In a sequential model,
downstream activations and routing naturally respond to the earlier ablation.
Missing ready-boundary checkpoints remain missing; this module never trains.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from .source_routed_ring_training import load_module, routed_modules, sha, state_digest, write


def zero_code_snapshot(served):
    """Intervene at the public generation callback / named mix boundary."""
    served.routing = deepcopy(served.routing)
    routing = served.routing
    if routing.model_forward is None:
        original = routing.generate
        def generate(models, context, candidate, weights):
            return original(models, context, replace(candidate, codes=torch.zeros_like(candidate.codes)), weights)
        routing.generate = generate
        sites = ["single"]
    else:
        original = routing.model_forward
        class ZeroMixedCodes:
            def __init__(self, execution):
                self.execution = execution
            def mix(self, site_name, logits):
                codes = self.execution.mix(site_name, logits)
                return torch.zeros_like(codes)
        def forward(models, context, candidate, execution):
            return original(models, context, candidate, ZeroMixedCodes(execution))
        routing.model_forward = forward
        sites = list(routing.sites)
    return dict(sites=sites, table_zeroed=False, parameters_modified=False,
                intervention="Zero returned mixed generation codes after native routing; sequential downstream routing may respond to the changed hidden activation")


@torch.no_grad()
def endpoint(report_path, quality):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    result = dict(fixture=report["fixture"], catalog_id=report["catalog_id"],
                  training_receipt=dict(path=str(report_path), sha256=sha(report_path)),
                  zero_training=True, heldout_not_a_training_signal=True)
    checkpoint = report.get("checkpoint")
    if checkpoint is None:
        result.update(status="NOT_MEASURED", reason=report.get("checkpoint_error", "No saved endpoint checkpoint"),
                      missing_artifact="Complete policy/models/optimizer/RNG checkpoint at the last completed update boundary, matching the held-out target and served selection")
        return result
    if sha(checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("trained endpoint checkpoint differs from source receipt")
    module, make_loop, update, evaluate, save, neutral = routed_modules(report["fixture"])
    with torch.random.fork_rng(devices=[]):
        loop = make_loop()
        saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
        loop.policy.load_state_dict(saved["policy"])
        for key in ("data_rng", "paired_noise_rng"):
            if key in saved:
                getattr(loop, key).set_state(saved[key])
        loop.initial_rmse = saved["initial_rmse"]
        data = np.load(report["observations"]["path"])
        if int(data["steps"][-1]) != loop.policy.completed_steps:
            raise ValueError("held-out capture and checkpoint do not share an endpoint")
        # A moving target is task-owned state absent from the example's policy
        # checkpoint. Bind its exact terminal held-out target to the saved NPZ.
        loop.test_targets.copy_(torch.from_numpy(data["target"][-1]))
        before = state_digest(save(loop))
        live_served = loop.policy.served_model()
        live = live_served.routed_forward(loop.test_context, perturb=False, output_noise=False)
        zero_served = loop.policy.served_model()
        result["intervention"] = zero_code_snapshot(zero_served)
        zero = zero_served.routed_forward(loop.test_context, perturb=False, output_noise=False)
        target = loop.test_targets
        neutral_values = neutral(loop)
        result["live_correspondence"] = quality.paired_edit_metrics(live.cpu().numpy(), target.cpu().numpy(), neutral_values.cpu().numpy())
        result["zero_code_correspondence"] = quality.paired_edit_metrics(zero.cpu().numpy(), target.cpu().numpy(), neutral_values.cpu().numpy())
        live_mse = float((live - target).square().mean())
        zero_mse = float((zero - target).square().mean())
        result.update(step=loop.policy.completed_steps, sampling_policy="All source held-out contexts/tokens; clean deterministic selected served bank, perturb=False/output_noise=False",
                      live_mse=live_mse, zero_code_mse=zero_mse, zero_code_minus_live_mse=zero_mse - live_mse,
                      useful_code_mse_pass=bool(zero_mse - live_mse > 1e-6))
        # Freeze the same learned-game judge and paired base noises for both
        # interventions. This judge is additional endpoint evidence only: it
        # does not recreate the stochastic DV12 training-game expectation.
        critic = live_served.critic
        private = torch.Generator(device="cpu").manual_seed(902)
        loss = loop.policy.recipe.make_loss()
        live_losses, zero_losses = [], []
        for _ in range(4):
            real = live_served.output_sigma * torch.randn(target.shape, generator=private)
            real_logits = critic(real)
            live_losses.append(float(loss.g_loss(critic(real + (live - target) / critic.scale), real_logits)))
            zero_losses.append(float(loss.g_loss(critic(real + (zero - target) / critic.scale), real_logits)))
        judge = dict(scope="Frozen served critic/RpGAN generator loss; 4 matched private paired-output-noise panels; clean held-out generation without DV12; not a full training-game gradient claim",
                     private_noise_seed=902, sigma=live_served.output_sigma,
                     live_losses=live_losses, zero_code_losses=zero_losses,
                     zero_code_minus_live_loss=[z - l for l, z in zip(live_losses, zero_losses)])
        result["fixed_learned_game_judge"] = judge
        useful = quality.useful_code_metrics([float(np.mean(live_losses))], [float(np.mean(zero_losses))])
        result["useful_code_judge_gate"] = useful
        after = state_digest(save(loop))
        result["training_state_purity"] = dict(before_sha256=before, after_sha256=after, exact=before == after)
        result["captured_prediction_exact"] = bool(np.array_equal(live.cpu().numpy(), data["prediction"][-1]))
        if before != after or not result["captured_prediction_exact"]:
            raise RuntimeError("read-only ablation lost the exact trained endpoint")
        result["status"] = "PASS" if result["useful_code_mse_pass"] and useful["passed"] else "FAIL"
        result["full_added_endpoint_gate"] = bool(result["live_correspondence"]["passed"] and result["status"] == "PASS")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--quality-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    quality = load_module(args.quality_module, "audit_ablation_quality")
    result = endpoint(args.receipt, quality)
    result["source_binding"] = dict(ablation_sha256=sha(__file__), quality_sha256=sha(args.quality_module),
                                    quality_version=quality.VERSION)
    write(args.output, result)
    print(json.dumps({"fixture": result["fixture"], "status": result["status"],
                      "delta_mse": result.get("zero_code_minus_live_mse")}), flush=True)


if __name__ == "__main__":
    main()
