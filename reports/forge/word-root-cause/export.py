"""Publish exact task-only Recipes and GIF receipts from completed observations.

This performs no training, optimizer updates, evaluation draws, or qualification.
Run with the documented Python 3.12 renderer environment.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
from PIL import Image, __version__ as pillow_version
import torch

from particlegan import Recipe
from experiments.forge.contracts import stable_hash
from experiments.forge.techniques import technique_signature
from experiments.forge.word_adapter import word_context


def identity(path):
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    report = ROOT / "reports/forge/word-root-cause"
    chosen = [("k3p", 3, "k3p-coeff170-cap1"), ("ka2", 2, "ka2-slow-prior-0p1"),
              ("r1r2", 1, "r1r2-mid-rate-fast-roles")]
    exports, media = [], []
    for family, round_number, name in chosen:
        path = report / "receipts" / f"{name}.json"
        receipt = json.loads(path.read_text())
        raw = Path(receipt["raw_directory"])
        recipe = Recipe(**receipt["recipe"])
        full = json.loads(json.dumps(asdict(recipe)))
        assert json.loads(json.dumps(recipe.to_dict())) == receipt["recipe"]
        assert all(full[key] == field["value"] for key, field in
                   receipt["field_ownership"]["recipe_fields"].items())
        convergence = receipt["grade"]["evaluator_result"]["convergence"]
        assert receipt["grade"]["gate_status"] == "PASS" and convergence["passing_suffix"] >= 5
        parent_path = ROOT / receipt["parent"]
        assert identity(parent_path)["sha256"] == receipt["parent_sha256"]
        parent = json.loads(parent_path.read_text())
        original = word_context({"candidate": parent, "protocol": {"seed": 0}},
                                receipt["task"], "cpu", root=ROOT).recipe
        before, after = technique_signature(original), technique_signature(recipe)
        transitions = {key: {"parent": before["mechanisms"][key], "executed": value}
                       for key, value in after["mechanisms"].items()
                       if before["mechanisms"][key] != value}
        gif = report / "media" / f"{family}.gif"
        gif.parent.mkdir(exist_ok=True)
        shutil.copyfile(raw / "goal.gif", gif)
        records = torch.load(raw / "observed-records.pt", map_location="cpu", weights_only=False)
        indices = np.rint(np.linspace(0, len(records) - 1, 9)).astype(int)
        with Image.open(gif) as im:
            frames = im.n_frames
        media.append({"family": family, "arm": name, "path": str(gif.relative_to(ROOT)),
                      **identity(gif), "frames": frames,
                      "stored_observations": receipt["raw_artifacts"]["observed-records.pt"],
                      "selected_record_indices": indices.tolist(),
                      "frame_updates": [records[index]["step"] for index in indices],
                      "source_receipt": {"path": str(path.relative_to(ROOT)), **identity(path)},
                      "renderer": {"python": platform.python_version(), "torch": str(torch.__version__),
                                   "matplotlib": matplotlib.__version__, "pillow": pillow_version},
                      "optimizer_updates": 0, "new_sampling_draws": 0,
                      "scope": "Nine frames selected from saved actual training observations; numerical grade uses all 24 checks."})
        exports.append({"family": family, "arm": name, "recipe": full,
                        "full_recipe_sha256": stable_hash(full),
                        "normalized_recipe_sha256": stable_hash(recipe.to_dict()),
                        "parent": {"path": receipt["parent"], "sha256": receipt["parent_sha256"]},
                        "receipt": {"path": str(path.relative_to(ROOT)), **identity(path)},
                        "source_commit": receipt["source_commit"], "source_digest": receipt["source_digest"],
                        "task_fingerprint": receipt["task_fingerprint"], "prior": receipt["prior"],
                        "initialization": receipt["initialization"], "runtime": receipt["runtime"],
                        "protocol_seed": 0, "protocol_round": round_number,
                        "rng_manifest_sha256": receipt["rng_manifest_sha256"],
                        "task": {"path": "configs/forge/tasks/five_word_joint_acquisition.json",
                                 "updates": 20001, "schedule_horizon": 20000,
                                 "observations": 24, "terminal_passes": 5,
                                 "sampling_law": receipt["sampling_law"]},
                        "technique_signature": after,
                        "activation_transitions_from_original_parent": transitions,
                        "final_metrics": receipt["final_metrics"], "convergence": convergence,
                        "qualification_input": False, "qualification_reuse": False,
                        "eligible_for_default": False})
    write(ROOT / "configs/forge/selections/word-joint-task-v1.json", {
        "schema_version": 1, "id": "word-joint-task-v1", "scope": "task_only_diagnostic",
        "qualification_input": False, "qualification_reuse": False, "eligible_for_default": False,
        "selection_rule": "Post-study task-only recommendations: coefficient-only K3P contrast, KA2 prior0.1 sustained pass, and R1/R2 mid-rate long sustained suffix; no cross-task or speed ranking.",
        "classification_note": "K3P/KA2 clean/full recipes cross the original parents' training noise, output warmup, and network cap activation boundaries. These are authorized existing-control ablation diagnostics, not ordinary same-signature Forge configuration_search outputs. Round2 prior-rate and round3 positive coefficient/cap contrasts preserve their actual measured clean/full comparator signatures. Frozen legacy numeric-knob labels are not ordinary search qualification.",
        "recipes": exports})
    write(report / "media/receipts.json", {"schema_version": 1, "media": media})
    print(json.dumps({"recipes": len(exports), "gifs": len(media), "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
