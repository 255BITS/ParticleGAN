import gzip, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sorted((Path(__file__).resolve().parent / "runs").glob("run-*"))[-1]
out_png = Path(__file__).resolve().parent / "chevron_wide_scatter.png"
means = np.array([[-8.0, 8.0], [8.0, 8.0], [0.0, -16.0]], dtype=np.float64)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
titles = [
    ("winner_published_absolute", "Winner (batchfeat absolute)"),
    ("control_mlp_matched_init", "Control (MLP matched init)"),
]
note = "schematic from mass counts"
rng = np.random.default_rng(0)
for ax, (arm, title) in zip(axes, titles):
    payload = json.loads(gzip.open(run / f"{arm}.json.gz").read())
    live = payload["result"]["live"]
    fakes = None
    for key in ("fakes", "samples", "final_samples"):
        if key in payload["result"]:
            fakes = np.asarray(payload["result"][key])
            note = "episode samples"
            break
    if fakes is None:
        counts = live["component_counts"]
        pts = []
        for i, c in enumerate(counts):
            if c <= 0:
                continue
            noise = rng.normal(size=(int(c), 2)) * 1.0
            pts.append(means[i] + noise)
        fakes = np.concatenate(pts, 0) if pts else means
    status = payload["verdict"]["status"]
    mass = [round(100 * m) for m in live["component_mass"]]
    ax.scatter(fakes[:, 0], fakes[:, 1], s=4, alpha=0.25, c="#1f77b4", linewidths=0)
    ax.scatter(means[:, 0], means[:, 1], s=80, c="red", marker="x", zorder=5, label="target means")
    order = [0, 2, 1]
    ax.plot(means[order, 0], means[order, 1], "r--", alpha=0.4, lw=1)
    ax.set_title(f"{title}\nstatus={status}  mass%={mass}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle(f"chevron_wide_d16 @ tip 510e005 — local sigma=1.0, base=16 ({note})")
fig.tight_layout()
fig.savefig(out_png, dpi=140)
print("WROTE", out_png, "from", run)
