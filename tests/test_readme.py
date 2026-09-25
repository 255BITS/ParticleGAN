"""The README's training loop and GANTrainer snippet run as written (on a tiny budget)."""
import re
from pathlib import Path

import torch

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", README, re.S)
SMALL = "total_steps=3, batch_size=32, num_particles=64"


def _shrink(code, original):
    assert original in code, original
    return code.replace(original, f"get_recipe({SMALL})")


def test_readme_loop_and_trainer_snippet_run():
    loop = next(b for b in BLOCKS if "make_critic_penalty(opt_d)" in b and "for step in range" in b)
    trainer = next(b for b in BLOCKS if "GANTrainer(get_recipe()" in b)
    torch.manual_seed(0)
    namespace = {"__name__": "__readme__"}
    exec(compile(_shrink(loop, "get_recipe(total_steps=2000)"), "README.md", "exec"), namespace)
    assert namespace["opt_d"].record.observed_steps == 3
    assert torch.isfinite(namespace["g_loss"]) and torch.isfinite(namespace["d_loss"])
    exec(compile(_shrink(trainer, "get_recipe()"), "README.md", "exec"), namespace)
    assert namespace["trainer"].completed_steps == 3
    assert namespace["samples"].shape == (1024, 2) and torch.isfinite(namespace["samples"]).all()
