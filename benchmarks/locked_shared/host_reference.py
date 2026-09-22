"""Verify the six additional extracted hosts against unchanged pinned source."""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import ModuleType
import sys
from unittest.mock import patch

import torch

from .baseline import Candidate, METRICS, run_toy, write_json
from .reference import COMMIT

HASHES = {
    "residual_student.py": "923dbde86d8f664c75aec18a2a2a833450b060a664f08f5861598ebc96ddaeeb",
    "unipolar.py": "8c639b81739ede7429071b06b898ef75a690c0a9d2f4358622db55f328bcccfa",
    "ae_gan_hold.py": "0582a67a9c03139a0eca162363824bc08ce831735b4463319716e849d3e161ad",
    "cover_leftover.py": "de576f93235df4476a4b69ea213d620d579630f78e0c1cca77727a6a79503181",
    "unused_token_hold.py": "d0f3d83c4e552662364615e66a3ed3b44026adbd10abc3b74f0c4368739bcd9c",
    "mid_scale_identity.py": "5df23bc4c749ffa161e3a4ee1c4358711ec0fbbeca1f5578beaa4856b8fcd3ce",
    "shared_trajectory.py": "a1f76626d43c3ef96933bedac6809c8b11a44c1f02c30399488a3b3e49f5476c",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/behavioral_baseline/extraction_parity.json"))
    args = parser.parse_args()
    folder = args.reference.resolve() / "conceptmod" / "toys"
    for filename, sha in HASHES.items():
        if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != sha:
            raise ValueError(f"reference changed: {filename}; use {COMMIT}")
    packages = {}
    for name, path in (("conceptmod", folder.parent), ("conceptmod.toys", folder)):
        module = ModuleType(name)
        module.__path__ = [str(path)]
        packages[name] = module
    torch.set_num_threads(1)
    report = {"conceptmod_commit": COMMIT, "source_sha256": HASHES, "torch": str(torch.__version__), "rows": []}
    with patch.dict(sys.modules, packages):
        for filename in HASHES:
            sys.modules.pop("conceptmod.toys." + Path(filename).stem, None)
        for name in (Path(f).stem for f in HASHES if f != "shared_trajectory.py"):
            print(f"START parity={name}", flush=True)
            original = importlib.import_module("conceptmod.toys." + name)
            if name == "residual_student":
                expected = original.train_locked()
            elif name == "ae_gan_hold":
                expected = original.train(original.locked_config())
            elif name == "cover_leftover":
                expected = original.fit_cover_leftover(original.CoverRecipe())
            elif name == "unused_token_hold":
                expected = original.train(original.locked_recipe())
            else:
                expected = original.run_arm("locked_rpgan" if name == "unipolar" else "locked")
            extracted = run_toy(name, Candidate("locked_shared"))
            actual = extracted["ema"] if name == "cover_leftover" else extracted["live"]
            comparisons = {key: {"reference": expected[key], "extracted": actual[key],
                                  "abs_diff": abs(expected[key] - actual[key]),
                                  "matches": math.isclose(expected[key], actual[key], rel_tol=1e-6, abs_tol=1e-7)}
                           for key, _, _ in METRICS[name]}
            ok = all(c["matches"] for c in comparisons.values())
            report["rows"].append({"toy": name, "matches": ok, "metrics": comparisons})
            write_json(args.output, report)
            print(f"DONE parity={name} matches={ok}", flush=True)
    return 0 if all(row["matches"] for row in report["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
