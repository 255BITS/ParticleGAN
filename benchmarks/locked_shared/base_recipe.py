"""Historical stock Recipe('gan') components/optimizers on the same ring host, seed 0."""

import json
from pathlib import Path

import torch

from particlegan import get_recipe
from .mode_hold import train_mode_hold


def main():
    torch.set_num_threads(1)
    path = Path("reports/locked_shared/base_recipe.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"torch": torch.__version__, "seed": 0,
              "host": "Original 8-mode ring, 96-wide host MLPs, original initialization/evaluation; stock recipe supplies prior, losses, optimizers, EMA and LR schedule.",
              "rows": []}
    for steps in (1200, 7000):
        recipe = get_recipe("gan_legacy", total_steps=steps).replace(name="gan")
        print(f"START stock ring recipe steps={steps} particles={recipe.num_particles}", flush=True)
        row = train_mode_hold(training_recipe=recipe, diagnostics=True)
        report["rows"].append({"recipe": recipe.to_dict(), "ring": row})
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
