"""Acquisition-margin diagnostics for cold ring traces (diagnostic only).

Scored: first host checkpoint with 8 modes and with a passing check (8 modes,
HQ >= .9). Per update (clean particles, no output noise): first update at
which every mode has a particle within the .21 HQ radius, and the longest run
of such updates before update 1000.
"""
import json
import sys

import torch


def margin(path):
    data = json.load(open(path))
    obs = data["observations"]
    first8 = next((o["step"] for o in obs if o["modes"] == 8), None)
    first_pass = next((o["step"] for o in obs if o["modes"] >= 8 and o["hq"] >= .9), None)
    X, M = torch.tensor(data["trace"]), torch.tensor(data["means"])
    covered = ((torch.cdist(X, M) < .21).any(1)).all(-1).tolist()
    first_cover = next((t + 1 for t, c in enumerate(covered) if c), None)
    best = run = 0
    for c in covered[:1000]:
        run = run + 1 if c else 0
        best = max(best, run)
    return dict(first_checkpoint_8_modes=first8, first_passing_checkpoint=first_pass,
                first_update_all_modes_covered=first_cover,
                longest_all_covered_run_before_1000=best,
                updates_all_covered_before_1000=sum(covered[:1000]),
                terminal_hq=[round(o["hq"], 3) for o in obs if o["step"] >= 1000],
                verdict=data["verdict"]["status"])


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path, json.dumps(margin(path)))
