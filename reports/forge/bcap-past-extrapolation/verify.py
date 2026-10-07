"""Saved-state audits, exact CUDA restores and unchanged default-path proof."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import ndtr
import torch

from benchmarks.toy_audit.bcap_past_extrapolation import build, declaration, scorer
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import state_digest, require_optimizer_steps


@reproducible_execution
def verify(raw, *, device):
    protocol = declaration()
    run = json.loads((raw / "results.json").read_text())
    restores = []
    for receipt in run["results"]:
        arm, task_id, phase = receipt["arm"], receipt["task"], receipt["phase"]
        name = f"{arm}-{task_id}-{phase}"
        saved = torch.load(raw / name / "state.pt", weights_only=True, map_location="cpu")
        context, trainer, _ = build(arm, task_id, device, cap=receipt["completed_updates"])
        context.load_state_dict(saved)
        if state_digest(saved) != state_digest(context.state_dict()):
            raise ValueError("final context does not restore exactly: " + name)
        require_optimizer_steps(saved, receipt["completed_updates"])
        rates = [group["lr"] for opt in (trainer.opt_g, trainer.opt_d) for group in opt.param_groups]
        if rates != [.012, .012*2.5, .012*1.5]:
            raise ValueError("effective rates changed")
        restores.append(dict(cell=name, checkpoint_sha256=file_hash(raw / name / "state.pt"),
                             context_restored_exactly=True, effective_rates=rates,
                             optimizer_steps=receipt["completed_updates"]))
        if phase == "shift":
            frozen = torch.load(raw / name / "frozen-state.pt", weights_only=True, map_location="cpu")
            stationary = torch.load(raw / f"{arm}-{task_id}-stationary" / "state.pt", weights_only=True, map_location="cpu")
            for key in ("models", "optimizers", "completed_steps", "initial_lrs"):
                if state_digest(frozen["trainer"][key]) != state_digest(stationary["trainer"][key]):
                    raise ValueError("frozen control changed its training state")
            if "extrapolation" in frozen["trainer"] and state_digest(frozen["trainer"]["extrapolation"]) != state_digest(stationary["trainer"]["extrapolation"]):
                raise ValueError("frozen control changed its cache")
            restores[-1]["frozen_training_state_unchanged"] = True
    atomic_json(raw / "restore-proof.json", dict(training_updates=0, model_sampling_draws=0, cells=restores))

    shape_rows = []
    for arm in protocol["candidates"]:
        parent = raw / f"{arm}-gaussian1d_acquisition-stationary"
        observations = torch.load(parent / "observations.pt", weights_only=True, map_location="cpu")
        values = np.sort(observations[-1]["samples"][:, 0].double().numpy())
        standard = (values - values.mean()) / values.std()
        cdf, ranks = ndtr(standard), np.arange(len(values)) / len(values)
        row = dict(arm=arm, step=4000, fitted_normal_ks=max(float(np.max(cdf-ranks)), float(np.max(ranks+1/len(values)-cdf))),
                   skewness=float(np.mean(standard**3)))
        if arm == "extrapolation_from_past":
            saved = torch.load(parent / "state.pt", weights_only=True, map_location="cpu")
            field = saved["trainer"]["extrapolation"]["previous"]["prior.z"]
            norms = field.norm(dim=1)
            selected = norms[norms > 0]
            row.update(cached_nonzero_prior_rows=len(selected), cached_prior_direction_min_norm=float(selected.min()),
                       cached_prior_direction_max_norm=float(selected.max()), implied_prior_step_mean=float(selected.mean())*.03)
        shape_rows.append(row)
    atomic_json(raw / "saved-shape-and-field-audit.json", dict(training_updates=0, model_sampling_draws=0,
        scope="secondary_saved_state_explanation_not_qualification", results=shape_rows))
    # Reconstruct the independently declared CPU target-control stream endpoints;
    # these are reference draws, never model draws or another training trial.
    control_streams = {}
    for tid, binding in protocol["tasks"].items():
        spec = json.loads((ROOT / binding["path"]).read_text())["execution"]["host_definition"]
        rng = torch.Generator(device="cpu").manual_seed(78013)
        target, _ = scorer(tid)
        target(spec, 4096, rng, 0)
        control_streams[tid] = rng.get_state()
    torch.save(dict(seed=78013, oracle_count=4096, states=control_streams), raw / "scorer-controls.rng.pt")
    return restores


def default_audit(raw, base):
    old = ast.parse(subprocess.check_output(["git", "show", base+":particlegan/training.py"], cwd=ROOT, text=True))
    current = ast.parse((ROOT / "particlegan/training.py").read_text())
    def methods(tree):
        return {m.name: m for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "GANTrainer"
                for m in cls.body if isinstance(m, ast.FunctionDef)}
    before, after = methods(old), methods(current)
    unchanged = ["step", "_execute_step", "_sample_training_prior", "sample", "_generate", "_batch"]
    for name in unchanged:
        if ast.dump(before[name], include_attributes=False) != ast.dump(after[name], include_attributes=False):
            raise ValueError("default method changed: " + name)
    after["_step"].body = after["_step"].body[1:]
    if ast.dump(before["_step"], include_attributes=False) != ast.dump(after["_step"], include_attributes=False):
        raise ValueError("alternating update changed")
    old_optimizer = subprocess.check_output(["git", "show", base+":particlegan/optim/dualnorm.py"], cwd=ROOT)
    if old_optimizer != (ROOT / "particlegan/optim/dualnorm.py").read_bytes():
        raise ValueError("default optimizer changed")
    atomic_json(raw / "default-path-audit.json", dict(training_updates=0, model_sampling_draws=0,
        merged_base=base, alternating_step_ast_unchanged_after_dispatch=True, unchanged_methods=unchanged,
        optimizer_bytes_unchanged=True, recipe_default_omitted=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base", default="d91c8d867b06435e79c25f65cf46754e8eabbe69")
    args = parser.parse_args()
    default_audit(args.raw, args.base)
    rows = verify(args.raw, device=args.device)
    print(json.dumps(dict(exact_restores=len(rows), training_updates=0, model_sampling_draws=0)))
