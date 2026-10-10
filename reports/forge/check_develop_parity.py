"""Bounded source-equivalence probe; it does not requalify scientific gates.

Run this file with the same Python/runtime and different PYTHONPATH values.
Every saved checkpoint field is compared, excluding reaction wall time only.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from particlegan import GANTrainer, ParticlePrior, Recipe


def digest(value):
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(repr(("tensor", str(tensor.dtype), tuple(tensor.shape))).encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b"dict")
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            result.update(repr((type(item).__name__, item)).encode())
        result.update(b"\0")
    visit(value)
    return result.hexdigest()


def state(trainer):
    saved = trainer.state_dict()
    last = saved.get("birth_death", {}).get("last", {})
    last.pop("eval_seconds", None)
    return saved


def build(config, serial):
    torch.manual_seed(314159)
    generator = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    critic = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    prior = ParticlePrior(config["num_particles"], 2,
                          generator=torch.Generator().manual_seed(314160))
    return GANTrainer(Recipe(**config), generator, critic, prior=prior,
                      seed=314159, serial_backward=serial)


def run(root, output, updates):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = json.loads((root / "tests/fixtures/feature-auto-base.json").read_text())
    base.update(num_particles=1024, z_dim=2, batch_size=128)
    cases = {}
    real = torch.arange(256, dtype=torch.float32).reshape(128, 2).sin()
    for name in ("e22", "atlas"):
        config = {**base}
        if name == "e22":
            config.update(birth_death_backend="knn", reopen_guard=None)
            for key in ("birth_death_cells", "birth_death_metric_rank", "birth_death_chunk", "birth_death_parent_policy"):
                config.pop(key, None)
        for serial in (False, True):
            label = f"{name}/serial={serial}"
            trainer = build(config, serial)
            initial = digest(state(trainer))
            trajectory = hashlib.sha256()
            resume = None
            for step in range(updates):
                trainer.step(real, collect_stats=True)
                trajectory.update(digest(state(trainer)).encode())
                if step == updates - 11:
                    resume = deepcopy(trainer.state_dict())
            final = digest(state(trainer))
            replica = build(config, serial)
            replica.load_state_dict(resume)
            for _ in range(10):
                replica.step(real, collect_stats=True)
            assert digest(state(replica)) == final, f"restart differs: {label}"
            sampling = {}
            for ema in (False, True):
                for noisy in (False, True):
                    stream = torch.Generator().manual_seed(31337)
                    sampling[f"ema={ema}/output_noise={noisy}"] = digest(
                        trainer.sample(128, ema=ema, output_noise=noisy, generator=stream))
            birth = trainer.birth_death.state_dict()
            cases[label] = {"initial": initial, "trajectory": trajectory.hexdigest(),
                            "final": final, "resume_exact": True, "sampling": sampling,
                            "backend": trainer.state_dict().get("backend_selection", {}).get("actual_backend", "knn"),
                            "penalty_calls": trainer.opt_d.record.calls,
                            "reaction_count": birth["counters"]["evals"],
                            "updates": updates}
            print(json.dumps({"completed": label, "updates": updates}), flush=True)
    output.write_text(json.dumps({"schema": 1, "scope": "fixed CPU API conformance; not scientific gate qualification",
                                 "excluded_diagnostics": ["birth_death.last.eval_seconds"], "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=840)
    args = parser.parse_args()
    if args.updates < 810:
        parser.error("probe must cross KA2's call-800 transition and include ten resumed updates")
    run(args.root, args.output, args.updates)
