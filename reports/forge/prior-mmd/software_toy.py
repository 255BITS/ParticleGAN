"""Declared public-API software fixture, not a Tier 2 qualification run.

Seed 0, 64 updates, 64 fixed-capacity 2D MoG locations, sigma .025, batch 64.
Both arms displace their initialized raw locations by +1.5 after reference
capture. BCAP leader versus exactly prior_regularizer=initial_mmd/prior_reg=10.
Gate: finite models, G/D actually update, active MMD <= .75 initial MMD and
active final MMD < disabled final MMD. Gaussian target N(0,I) is training data;
the tether only sees initialized raw locations. This bounded mechanism fixture
does not assert target quality or replace any Forge task/evaluation gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.api import FormulationContext
from experiments.forge.state import state_digest
from particlegan import InitialMMDRegularizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records, receipts = {}, {}
    for name, delta in (("leader", {}), ("initial_mmd", dict(prior_regularizer="initial_mmd", prior_reg=10.))):
        context = FormulationContext(recipe_preset="bcap", recipe_overrides=dict(
            num_particles=64, z_dim=2, batch_size=64, total_steps=64, **delta),
            prior=dict(kind="mog", sigma=.025, standardize=False), prior_contract="recipe_owned_v1", seed=0)
        g = context.construct(lambda: nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2)), component="generator")
        d = context.construct(lambda: nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)), component="discriminator")
        trainer = context.build_trainer(g, d)
        initial_models = {k: deepcopy(getattr(trainer, k).state_dict()) for k in ("G", "D")}
        initial = trainer.prior.z.detach().clone()
        score = InitialMMDRegularizer(initial)
        with torch.no_grad():
            trainer.prior.z.add_(1.5)
        data = context.streams.generator("data", component="software_toy", purpose="gaussian")
        batch_hash = hashlib.sha256()
        observations = []
        for step in range(65):
            if step % 8 == 0:
                with torch.no_grad():
                    value = float(score(trainer.prior.z))
                    sample = trainer.sample(256)
                observations.append(dict(update=step, mmd=value, locations=trainer.prior.z.detach().cpu().numpy().copy(),
                                         samples=sample.detach().cpu().numpy().copy()))
                print(json.dumps(dict(arm=name, update=step, mmd=value)), flush=True)
            if step < 64:
                target = torch.randn((64, 2), generator=data)
                batch_hash.update(target.numpy().tobytes())
                trainer.step(target)
        updated = {key: any(not torch.equal(value, initial_models[key][field])
                           for field, value in getattr(trainer, key).state_dict().items()) for key in ("G", "D")}
        finite = all(torch.isfinite(value).all().item() for module in (trainer.G, trainer.D, trainer.prior)
                     for value in module.state_dict().values() if isinstance(value, torch.Tensor))
        receipts[name] = dict(recipe=trainer.recipe.to_dict(), initial_locations_sha256=state_digest(initial),
            initial_networks_sha256=state_digest(initial_models), batches_sha256=batch_hash.hexdigest(),
            initial_mmd=observations[0]["mmd"], final_mmd=observations[-1]["mmd"], models_finite=finite,
            networks_updated=updated, completed_steps=64)
        records[name] = observations
        torch.save(context.state_dict(), args.artifacts / f"{name}.pt")
    assert receipts["leader"]["initial_locations_sha256"] == receipts["initial_mmd"]["initial_locations_sha256"]
    assert receipts["leader"]["initial_networks_sha256"] == receipts["initial_mmd"]["initial_networks_sha256"]
    assert receipts["leader"]["batches_sha256"] == receipts["initial_mmd"]["batches_sha256"]
    active, baseline = receipts["initial_mmd"], receipts["leader"]
    ratio = active["final_mmd"] / active["initial_mmd"]
    passed = bool(all(r["models_finite"] and all(r["networks_updated"].values()) for r in receipts.values())
                  and ratio <= .75 and active["final_mmd"] < baseline["final_mmd"])
    publication = ROOT / "reports/forge/prior-mmd"
    frames = []
    oracle = torch.randn((256, 2), generator=torch.Generator().manual_seed(9901)).numpy()
    for i in range(9):
        canvas = Image.new("RGB", (960, 560), "white")
        draw = ImageDraw.Draw(canvas)
        step = records["leader"][i]["update"]
        draw.text((18, 8), f"Actual public-API training, update {step}/64. Software fixture; no target-quality qualification.", fill="black")
        for col, name in enumerate(("leader", "initial_mmd")):
            obs = records[name][i]
            draw.text((col * 480 + 18, 32), f"{name}: MMD {obs['mmd']:.5f}", fill="black")
            for row, (title, desired, actual) in enumerate((("Raw prior: initial reference (gray), current (blue)", initial.numpy(), obs["locations"]),
                                                          ("Outputs: Gaussian target (gray), API samples (blue)", oracle, obs["samples"]))):
                x0, y0 = col * 480 + 50, row * 250 + 80
                draw.text((col * 480 + 18, y0 - 22), title, fill="black")
                draw.rectangle((x0, y0, x0 + 380, y0 + 210), outline="#777777")
                for points, color in ((desired, "#bbbbbb"), (actual, "#185ad8")):
                    for x, y in np.clip(points, -4, 4):
                        px, py = x0 + (float(x) + 4) / 8 * 380, y0 + (4 - float(y)) / 8 * 210
                        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
        frames.append(canvas)
    frames[0].save(publication / "goal.gif", save_all=True, append_images=frames[1:], duration=450, loop=0)
    result = dict(scope="software_mechanism_fixture_not_qualification", protocol_seed=0, status="PASS" if passed else "FAIL",
                  gates=dict(active_mmd_ratio=dict(score=ratio, target="<= .75"),
                      active_better_than_leader=dict(score=active["final_mmd"], target=f"< {baseline['final_mmd']}"),
                      models_finite=dict(score=all(r["models_finite"] for r in receipts.values()), target=True),
                      networks_update=dict(score=all(all(r["networks_updated"].values()) for r in receipts.values()), target=True)),
                  arms=receipts, artifacts=str(args.artifacts),
                  source_base_commit="0337bbbc2300fd351e77e9a222b92401137fd421",
                  source_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__), ROOT / "particlegan/initial_mmd.py", ROOT / "particlegan/recipes.py", ROOT / "particlegan/training.py")})
    (publication / "software-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], gates=result["gates"])), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
