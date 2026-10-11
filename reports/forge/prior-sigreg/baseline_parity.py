"""Four-update software parity probe; run unchanged on base and feature trees."""
import json
import torch
from torch import nn

from experiments.forge.api import FormulationContext
from experiments.forge.state import state_digest


torch.set_num_threads(1)
torch.manual_seed(777)  # fixed caller state; comparison protocol stays seed 0
context = FormulationContext(recipe_preset="bcap", seed=0,
    recipe_overrides={"num_particles": 12, "z_dim": 2, "batch_size": 4, "total_steps": 4},
    prior={"kind": "mog", "sigma": .025, "standardize": False},
    prior_contract="recipe_owned_v1")
generator = context.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)), component="generator")
critic = context.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)), component="discriminator")
trainer = context.build_trainer(generator, critic)
real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
for _ in range(4):
    trainer.step(real)
print(json.dumps({"trainer_state_sha256": state_digest(trainer.state_dict()),
                  "named_streams_sha256": state_digest(context.streams.state_dict())}))
