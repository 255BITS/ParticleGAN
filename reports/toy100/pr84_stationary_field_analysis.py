"""Offline critic-field slopes at exact saved PR84 continuation states.

This never enters training.  Ring centers are used solely to explain a saved
failure, and the saved accepted-D critic is scored with PR84's same stencil.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator


def smoothed_score(critic, points, width):
    values = [critic(points)]
    for dim in range(2):
        offset = torch.zeros_like(points)
        offset[:, dim] = width
        values += [critic(points + offset), critic(points - offset)]
    return torch.stack(values).mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    diagnosis = json.loads(args.diagnosis.read_text())
    if diagnosis["status"] != "EXACT_REFERENCE_PARITY":
        raise RuntimeError("capture failed exact reference parity")
    if hashlib.sha256(args.states.read_bytes()).hexdigest() != diagnosis["selected_states_sha256"]:
        raise RuntimeError("saved-state SHA differs from capture verdict")
    states = torch.load(args.states, weights_only=True)
    rows = {row["step"]: row for row in diagnosis["rows"]}
    means = mode_hold.ring_means()
    analyzed = []
    for step, phases in sorted(states.items()):
        row = rows[step]
        saved = phases["post_accepted_d"]["critic"]
        critic = SimpleMLPDiscriminator(2, mode_hold.HIDDEN,
                                       mode_hold.N_HIDDEN, mode_hold.FOURIER)
        critic.load_state_dict({name.removeprefix("model."): tensor
                                for name, tensor in saved.items()})
        critic.eval()
        points = torch.tensor(row["stages"]["pre_step"], dtype=torch.float32)
        accepted = torch.tensor(row["stages"]["bounded_joint"], dtype=torch.float32)
        unbounded = torch.tensor(row["stages"]["unbounded_joint"], dtype=torch.float32)
        network = torch.tensor(row["stages"]["bounded_network_only"], dtype=torch.float32)
        prior = torch.tensor(row["stages"]["bounded_prior_only"], dtype=torch.float32)
        width = row["stages"]["record"]["critic_width"]
        x = points.detach().clone().requires_grad_(True)
        score = smoothed_score(critic, x, width)
        gradient = torch.autograd.grad(score.sum(), x)[0].detach()
        nearest = torch.cdist(points, means).argmin(dim=1)
        to_center = means[nearest] - points
        unit_to_center = to_center / to_center.norm(dim=1, keepdim=True).clamp_min(1e-12)
        delta = accepted - points
        net_delta = network - points
        prior_delta = prior - points
        with torch.no_grad():
            score_after = smoothed_score(critic, accepted, width)
            score_unbounded = smoothed_score(critic, unbounded, width)
        analyzed.append(dict(
            step=step, width=width,
            exact_live=row["grades"]["bounded_joint"],
            record=row["stages"]["record"],
            particles=[dict(
                index=i, nearest_mode=int(nearest[i]),
                center_distance_before=float(to_center[i].norm()),
                center_distance_after=float((means[nearest[i]] - accepted[i]).norm()),
                score_before=float(score[i].detach()), score_after=float(score_after[i]),
                score_unbounded=float(score_unbounded[i]),
                score_gradient=gradient[i].tolist(),
                score_slope_toward_center=float((gradient[i] * unit_to_center[i]).sum()),
                score_gradient_dot_actual_move=float((gradient[i] * delta[i]).sum()),
                actual_move=delta[i].tolist(),
                network_only_move=net_delta[i].tolist(),
                prior_only_move=prior_delta[i].tolist(),
            ) for i in range(len(points))],
        ))
    result = dict(scope="offline_saved_accepted_D_critic_field",
                  source_capture_sha256=hashlib.sha256(args.diagnosis.read_bytes()).hexdigest(),
                  states_sha256=hashlib.sha256(args.states.read_bytes()).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=analyzed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps(dict(event="FIELD_ANALYSIS_DONE", steps=list(states))), flush=True)


if __name__ == "__main__":
    main()
