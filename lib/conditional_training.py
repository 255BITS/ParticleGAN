"""Shared conditional GAN/DDGAN optimizer loop; task-specific batches stay outside."""
import json
import time
import torch
from torch.nn import functional as F
from lib.trajectory import TrajectoryCritic

def run_updates(cfg, out, g, d, prior, noise, ema_g, ema_prior, ema_noise,
                opt_g, opt_d, gan, vic, reg, batch, fake, rngs, total_start):
    with (out / "metrics.jsonl").open("w") as log:
        for step in range(1, cfg["steps"]+1):
            d.requires_grad_(True)
            c, context, real, xt, t = batch()
            with torch.no_grad():
                xf, _ = fake(c, context, xt, t)
            dr, cr = d(real, c, context, xt, t)
            df, cf = d(xf, c, context, xt, t)
            ld = gan.d_loss(dr, df)
            if cfg["d_mode"] == "ucd" and cfg["ucd_lambda"]:
                target = d.ucd_labels(c, t)
                ld = ld + cfg["ucd_lambda"] * (F.cross_entropy(cr, target) + F.cross_entropy(cf, target))
            penalty, _ = reg.penalty(TrajectoryCritic(d, c, context, xt, t), real, xf, step, rngs[5], collect_stats=False)
            ld = ld + penalty
            opt_d.zero_grad(set_to_none=True)
            ld.backward()
            opt_d.step()
            d.requires_grad_(False)
            c, context, real, xt, t = batch()
            xf, ids = fake(c, context, xt, t)
            df = d(xf, c, context, xt, t)[0]
            with torch.no_grad():
                dr = d(real, c, context, xt, t)[0]
            lg = gan.g_loss(df, dr)
            if prior.kind == "learned" and cfg["prior_reg"]:
                selected = prior.table[ids.unique()]
                if len(selected) > 1:
                    lg = lg + cfg["prior_reg"] * vic(selected)
            opt_g.zero_grad(set_to_none=True)
            lg.backward()
            opt_g.step()
            with torch.no_grad():
                for target, source in ((ema_g, g), (ema_prior, prior), (ema_noise, noise)):
                    for pe, p in zip(target.parameters(), source.parameters()):
                        pe.lerp_(p, 1-cfg["ema"])
            if step % cfg["log_interval"] == 0 or step == cfg["steps"]:
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-total_start
                row = dict(step=step, d_loss=float(ld.detach()), g_loss=float(lg.detach()), train_seconds=elapsed,
                           samples_per_second=step*cfg["batch_size"]/elapsed)
                log.write(json.dumps(row, allow_nan=False)+"\n")
                log.flush()
                print(f"step={step}/{cfg['steps']} D={row['d_loss']:.4f} G={row['g_loss']:.4f} samples/s={row['samples_per_second']:.0f} train_s={elapsed:.1f}", flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter()-total_start
    return train_seconds
