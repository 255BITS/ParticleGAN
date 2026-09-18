#!/usr/bin/env python
"""Matched encoder-only fine-tuning from a frozen bounded scout checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_mog_oracle import center_errors, usage
from experiments.train_mog_autoencoder import RoutingEncoder, draw_rng, json_write
from lib.toy_models import SimpleMLPGenerator, sample_100gaussians
from particlegan import MoGParticlePrior

ARMS = ("oracle_query", "recon_st")


def query(encoder, x):
    scaled = x / math.sqrt(8.25 + .03**2)
    return scaled + encoder.net(scaled)[:, :2]


def fit_loss(arm, encoder, decoder, means, sigma, decoded, x, temperature):
    if arm == "oracle_query":
        with torch.no_grad():
            best, _, _ = center_errors(x, decoded)
            target = means[best].detach()
        return (query(encoder, x) - target).square().mean()
    if arm == "recon_st":
        z, _, _ = encoder(x, means, sigma, "route_zero", torch.zeros_like(x), temperature)
        return (decoder(z) - x).square().mean()
    raise ValueError(arm)


def state_hash(*models):
    digest = hashlib.sha256()
    for model in models:
        for key, value in model.state_dict().items():
            digest.update(key.encode())
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(encoder, means, decoded, real, best, best_error, baseline):
    chosen = torch.cat([((query(encoder, x)[:, None] - means[None]).square().sum(-1)).argmin(1)
                        for x in real.split(4096)])
    errors = (real.double() - decoded[chosen].double()).square().mean(-1)
    assert (errors >= best_error - 1e-12).all()
    mse, floor = float(errors.mean()), float(best_error.mean())
    result = dict(encoder_mse=mse, oracle_mse=floor, selection_gap=mse-floor,
                  gap_closed_fraction=(baseline-mse)/(baseline-floor),
                  improvement_fraction=(baseline-mse)/baseline,
                  id_agreement=float((chosen == best).double().mean()),
                  same_grid_mode=float((((decoded[chosen]+4.5).round().clamp(0, 9)) ==
                                        ((real+4.5).round().clamp(0, 9))).all(1).double().mean()),
                  usage=usage(chosen, len(means)), samples=len(real))
    return result


def load_models(run, device):
    cfg = json.loads((run / "config.json").read_text())
    saved = torch.load(run / "checkpoint.pt", map_location=device, weights_only=True)
    assert cfg["arm"] == "route_bounded" and saved["step"] == cfg["steps"] == 6000
    assert hashlib.sha256((run / "source.py").read_bytes()).hexdigest() == cfg["source_sha256"]
    torch.manual_seed(cfg["seed"])
    prior = MoGParticlePrior(num_particles=cfg["num_particles"], z_dim=cfg["z_dim"], device=device).to(device)
    decoder = SimpleMLPGenerator(cfg["z_dim"], cfg["width"]).to(device)
    encoder = RoutingEncoder(cfg["width"]).to(device)
    for model, key in ((prior, "prior"), (decoder, "g"), (encoder, "encoder")):
        model.load_state_dict(saved[key])
    prior.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)
    assert float(prior.sigma) == cfg["sigma"]
    return cfg, saved, prior, decoder, encoder


def train(arm, args):
    out = args.out / arm
    out.mkdir(parents=True, exist_ok=False)
    base, saved, prior, decoder, encoder = load_models(args.base_run, args.device)
    base_hash = hashlib.sha256((args.base_run / "checkpoint.pt").read_bytes()).hexdigest()
    frozen_hash = state_hash(prior, decoder)
    means = prior.means().detach()
    with torch.no_grad():
        decoded = decoder(means)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=base["lr_e"], betas=(0., .999))
    data_rng = draw_rng(args.device, base["seed"] + 2)
    data_rng.set_state(saved["data_rng"].cpu())
    real = sample_100gaussians(base["eval_samples"], args.device,
                              generator=draw_rng(args.device, base["seed"] + 10000))
    best_ids, best_errors, initial_errors = [], [], []
    with torch.no_grad():
        for x in real.split(4096):
            _, chosen, _ = encoder(x, means, prior.sigma, "route_zero", torch.zeros_like(x), base["temperature"])
            best, error, selected = center_errors(x, decoded, chosen)
            best_ids.append(best)
            best_errors.append(error)
            initial_errors.append(selected)
    best, best_error = torch.cat(best_ids), torch.cat(best_errors)
    initial = torch.cat(initial_errors)
    baseline = float(initial.mean())
    old_metrics = json.loads((args.base_run / "metrics.json").read_text())
    assert abs(float(initial[:8192].mean()) - old_metrics["zero_offset_mse"]) < 1e-7
    metadata = dict(arm=arm, steps=args.steps, batch_size=base["batch_size"], seed=base["seed"],
                    lr=base["lr_e"], betas=[0., .999], optimizer="fresh Adam; reset moments in both arms",
                    data_stream="resume base checkpoint data_rng; same state in both arms",
                    evaluation_seed=base["seed"]+10000, eval_samples=len(real),
                    temperature=base["temperature"], offset="zero", base_run=str(args.base_run),
                    base_checkpoint_sha256=base_hash, frozen_state_sha256=frozen_hash,
                    initial_encoder_sha256=state_hash(encoder), baseline_mse=baseline,
                    oracle_mse=float(best_error.mean()), sigma=float(prior.sigma),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    torch=torch.__version__, device=args.device)
    json_write(out / "config.json", metadata)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    train_seconds = 0.
    with (out / "log.txt").open("w", buffering=1) as log, (out / "history.jsonl").open("w", buffering=1) as history:
        def emit(s):
            print(s, flush=True)
            log.write(s + "\n")
        def record(step):
            assert state_hash(prior, decoder) == frozen_hash, "frozen model changed"
            metrics = evaluate(encoder, means, decoded, real, best, best_error, baseline)
            metrics.update(arm=arm, step=step, train_seconds=train_seconds)
            json_write(out / "metrics.json", metrics)
            history.write(json.dumps(metrics, allow_nan=False) + "\n")
            emit(f"EVAL {arm} step={step} mse={metrics['encoder_mse']:.8f} gap_closed={metrics['gap_closed_fraction']:.2%} agreement={metrics['id_agreement']:.2%}")
        emit(f"START {arm} updates={args.steps} fixed_sigma={float(prior.sigma):.8g}")
        record(0)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(1, args.steps+1):
            x = sample_100gaussians(base["batch_size"], args.device, generator=data_rng)
            optimizer.zero_grad(set_to_none=True)
            loss = fit_loss(arm, encoder, decoder, means, prior.sigma, decoded, x, base["temperature"])
            loss.backward()
            optimizer.step()
            if step == 1 or step % 250 == 0:
                emit(f"{arm} step={step}/{args.steps} objective={float(loss.detach()):.8f}")
            if step % 2000 == 0 or step == args.steps:
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                train_seconds += time.perf_counter()-started
                record(step)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
        assert prior.z.grad is None and all(p.grad is None for p in decoder.parameters())
        assert hashlib.sha256((args.base_run / "checkpoint.pt").read_bytes()).hexdigest() == base_hash
        torch.save(dict(encoder=encoder.state_dict(), opt=optimizer.state_dict(),
                        data_rng=data_rng.get_state(), step=args.steps,
                        base_checkpoint_sha256=base_hash), out / "checkpoint.pt")
        emit(f"DONE {arm} seconds={train_seconds:.2f}")


def export(run_dir, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report.mkdir(parents=True, exist_ok=True)
    configs, histories = {}, {}
    for arm in ARMS:
        run = run_dir / arm
        configs[arm] = json.loads((run / "config.json").read_text())
        histories[arm] = [json.loads(s) for s in (run / "history.jsonl").read_text().splitlines()]
        assert hashlib.sha256((run / "source.py").read_bytes()).hexdigest() == configs[arm]["source_sha256"]
        assert histories[arm][-1]["step"] == configs[arm]["steps"]
    shared = [{k: v for k, v in cfg.items() if k != "arm"} for cfg in configs.values()]
    assert shared[0] == shared[1], "unmatched experiment configurations"
    rows = sorted((h[-1] for h in histories.values()), key=lambda r: r["encoder_mse"])
    json_write(report / "results.json", dict(configs=configs, histories=histories, leaderboard=rows))
    baseline = configs[ARMS[0]]["baseline_mse"]
    floor = configs[ARMS[0]]["oracle_mse"]
    lines = ["# Frozen encoder fitting leaderboard", "",
             f"Start: bounded scout checkpoint, zero-offset MSE {baseline:.8f}; exhaustive center oracle {floor:.8f}. Same 100k held-out examples for every evaluation. Final weights only.", "",
             "| Arm | Updates | Hard-choice MSE ↓ | MSE reduction % | Oracle gap closed % | Oracle ID agreement % | Effective particles | Train sec |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['step']} | {r['encoder_mse']:.8f} | {100*r['improvement_fraction']:.2f} | {100*r['gap_closed_fraction']:.2f} | {100*r['id_agreement']:.2f} | {r['usage']['effective']:.1f} | {r['train_seconds']:.2f} |")
    lines += ["", "Gap closed = (initial MSE − final MSE) / (initial MSE − oracle MSE); negative values mean regression. G, particle means, and sigma are unchanged, so unconditional generation is unchanged by construction.", "",
              "![Learning curves](learning_curves.png)", ""]
    (report / "LEADERBOARD.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, history in histories.items():
        axes[0].plot([r["step"] for r in history], [r["encoder_mse"] for r in history], marker="o", label=arm)
        axes[1].plot([r["step"] for r in history], [100*r["id_agreement"] for r in history], marker="o", label=arm)
    axes[0].axhline(floor, color="black", ls="--", label="center oracle")
    axes[0].set(ylabel="Held-out zero-offset MSE")
    axes[1].set(ylabel="Oracle ID agreement (%)")
    for ax in axes:
        ax.set(xlabel="Encoder-only updates")
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(report / "learning_curves.png", dpi=160)
    plt.close(fig)
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, default=ROOT / "runs/mog_autoencoder/scout/route_bounded")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/mog_autoencoder/encoder_fit")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/mog-autoencoder/encoder-fit")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(2)
    for arm in ARMS:
        train(arm, args)
    export(args.out, args.report)


if __name__ == "__main__":
    main()
