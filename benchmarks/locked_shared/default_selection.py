"""Compare the historical stock recipe and R1+R2 on the ring, at stock capacity/budget.

This comparison isolates the penalty in the full recipe. It is separate from
the twelve-particle regression suite and does not claim 100-Gaussian results.
"""

from dataclasses import asdict
from pathlib import Path
import time

import torch

from particlegan import get_recipe
from .baseline import digest, protocol, write_json
from .mode_hold import train_mode_hold


def main():
    torch.set_num_threads(1)
    output = Path("reports/behavioral_baseline/stock_ring.json")
    if output.exists():
        raise FileExistsError(output)
    fingerprint = protocol()
    report = {"protocol": fingerprint, "protocol_sha256": digest(fingerprint), "seed": 0,
              "host": "8-mode ring, 96-wide MLPs, Fourier-3 critic; full stock prior/optimizer/schedule",
              "rows": []}
    for name, options in (("stock_b_cap", {}), ("stock_r1_r2_0_1", {"reg_arm": "a_r1r2", "reg_coeff": 0.1})):
        recipe = get_recipe("gan_legacy", **options).replace(name="gan")
        print(f"START {name} steps={recipe.total_steps} particles={recipe.num_particles}", flush=True)
        start = time.monotonic()
        def log(point):
            print(f"STEP {name} {point['step']}/{recipe.total_steps} live_modes={point['modes']}/8 live_HQ={point['hq']:.6f}", flush=True)
        result = train_mode_hold(training_recipe=recipe, diagnostics=True, log=log)
        report["rows"].append({"name": name, "recipe": asdict(recipe), "ring": result,
                               "seconds": time.monotonic() - start})
        write_json(output, report)
        live = result["live"]
        print(f"DONE {name} live_modes={live['modes']}/8 live_HQ={live['hq']:.6f} EMA_modes={result['modes']}/8 EMA_HQ={result['hq']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
