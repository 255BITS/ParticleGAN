"""The README's published GANTrainer example runs as written on a tiny budget."""
import re
from pathlib import Path

import torch
from particlegan import GANTrainer

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", README, re.S)
SMALL = "total_steps=3, batch_size=32, num_particles=64"


def _shrink(code, original):
    assert original in code, original
    return code.replace(original, f"get_recipe({SMALL})")


def test_readme_loop_and_trainer_snippet_run(monkeypatch):
    trainer = next(b for b in BLOCKS if "GANTrainer(recipe, G, D, prior=prior" in b)
    observed = []
    original_step = GANTrainer.step

    def record_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(GANTrainer, "step", record_step)
    torch.manual_seed(0)
    namespace = {"__name__": "__readme__"}
    exec(compile(_shrink(trainer, "get_recipe(total_steps=2000)"), "README.md", "exec"), namespace)
    assert namespace["trainer"].completed_steps == 3
    assert namespace["trainer"].opt_d.record.observed_steps == 3
    assert len(observed) == 3
    assert all(torch.isfinite(row[key]) for row in observed for key in ("loss_g", "loss_d"))
    assert namespace["samples"].shape == (1024, 2) and torch.isfinite(namespace["samples"]).all()
