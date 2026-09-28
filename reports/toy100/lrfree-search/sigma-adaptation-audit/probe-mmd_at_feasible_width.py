"""Two-point continuation of the read-only MMD sigma derivative probe."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import evaluate as probe

HERE = Path(__file__).resolve().parent
WIDTHS = (0.017, 0.019)


def main():
    started = time.monotonic()
    first = json.loads((HERE / "result.json").read_text())
    h = first["bandwidth"]["value"]
    sigma_orig = first["sigma"]
    state = torch.load(probe.RUN / "final-state.pt", map_location="cpu", weights_only=False)["trainer"]
    with np.load(probe.RUN / "native-clean/holdout_samples.npz") as a:
        clean = a["live"].astype(np.float64)
        real = a["target"].astype(np.float64)
    with np.load(probe.RUN / "native-noisy/holdout_samples.npz") as a:
        eps = (a["live"].astype(np.float64) - clean) / sigma_orig
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    critic = probe.SimpleMLPDiscriminator(in_dim=2, hidden_dim=128, n_hidden=3, fourier=3).to(device)
    critic.load_state_dict(state["models"]["D"])
    critic.eval().requires_grad_(False)
    reports = []
    for sigma in WIDTHS:
        probe.CURRENT_SIGMA = sigma
        rows = []
        for batch in range(probe.N_BATCHES):
            sl = slice(batch * probe.B, (batch + 1) * probe.B)
            x = clean[sl] + sigma * eps[sl]
            row = probe._pair_terms(x, eps[sl], real[sl], h)
            row["gan"] = probe._gan_gradient(critic, clean[sl], eps[sl], real[sl], sigma, device)
            row["batch"] = batch
            rows.append(row)
        reports.append(dict(sigma=sigma, mmd2=probe._summary(rows, "mmd2"),
                            mmd_gradient=probe._summary(rows, "grad_logsigma"),
                            gan_gradient=probe._summary([r["gan"] for r in rows], "grad_logsigma"),
                            rows=rows))
    result = dict(method="same held-out pool and calibration as evaluate.py; only applied sigma varied",
                  diagnostic_only=True, base_sigma=sigma_orig, bandwidth=h,
                  widths=reports, seconds=time.monotonic() - started)
    (HERE / "mmd_at_feasible_width.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"widths": [{k: v for k, v in r.items() if k != "rows"} for r in reports],
                      "seconds": result["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
