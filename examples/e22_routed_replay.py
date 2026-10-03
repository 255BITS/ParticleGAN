"""Activation-checkpoint the complete two-site E22 model and replay its DV12.

python -u examples/e22_routed_replay.py --steps 8
python -u examples/e22_routed_replay.py --steps 8 --mode no_rows --device cuda

Only the generator model forward is checkpointed. Policy observations, shared
paired noise, optimizer steps, row moves and averages remain in the outer loop.
"""

import argparse
from contextlib import nullcontext
import json

import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from e22_routed_sites import INITIALIZATIONS, MODES, PENALTY_UNITS, diagnostics, evaluate, make_loop, update


def checkpointed_generate(policy, context):
    """Replay a complete model with fresh routing and a private DV12 stream.

    Call after begin_step, and complete backward before any optimizer/controller
    update, row move or module-mode change. The model must not mutate buffers in
    its forward. The original call records DV12 normally; recomputation uses a
    private stream so it neither advances training RNG nor records diagnostics.
    Output noise belongs to the outer paired-error loop, outside this closure.
    """
    initial_rng = policy.noise_generator.get_state().clone()

    class Recompute:
        # This context is reusable: each backward replay starts at the same
        # state, without ever rewinding or replacing the live training stream.
        stream = None

        def __enter__(self):
            self.stream = torch.Generator(device=policy.device)
            self.stream.set_state(initial_rng)
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.stream = None

    replay = Recompute()

    def forward(inputs):
        # stream=None is the original training forward. The replay stream is
        # not policy.noise_generator, so routed_generate disables DV12 records.
        # Every call creates and closes a fresh RoutedExecution for both sites.
        return policy.routed_generate(inputs, sigma=0, perturb=True, stream=replay.stream)

    # Recompute the entire model, including the final routing site. PyTorch
    # also preserves default RNGs (e.g. dropout); our explicit Generator needs
    # the private replay above. No lifecycle hooks belong inside forward().
    with set_checkpoint_early_stop(False):
        return checkpoint(forward, context, use_reentrant=False,
                          context_fn=lambda: (nullcontext(), replay), preserve_rng_state=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initialization", choices=INITIALIZATIONS, default="api")
    parser.add_argument("--penalty-units", choices=PENALTY_UNITS, default="token")
    parser.add_argument("--max-context-harm", type=float)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(1)
    loop = make_loop(mode=args.mode, device=args.device, initialization=args.initialization,
                     penalty_units=args.penalty_units, max_context_harm=args.max_context_harm)
    for _ in range(args.steps):
        with torch.autograd.set_multithreading_enabled(False):
            row = update(loop, generator_forward=checkpointed_generate)
        print(json.dumps({"event": "train", "mode": args.mode, "step": row["step"],
                          "penalty_units": loop.config["penalty_units"],
                          "max_context_harm": loop.config["max_context_harm"],
                          "loss_g": row["loss_g"], "row_diagnostics": diagnostics(loop.policy)}), flush=True)
    print(json.dumps({"event": "complete", "mode": args.mode, **evaluate(loop),
                      "penalty_units": loop.config["penalty_units"],
                      "max_context_harm": loop.config["max_context_harm"],
                      "row_diagnostics": diagnostics(loop.policy)}), flush=True)


if __name__ == "__main__":
    main()
