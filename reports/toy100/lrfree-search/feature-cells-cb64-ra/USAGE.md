# ParticleGAN feature cell configuration

The [CB64-RA config](configs/overrides-CB64-RA.json) combines rank 8 feature cells with bounded real-anchor parent selection. It requires the matching [candidate package](pkg-CB64-RA/particlegan/training.py). The [experiment report](REPORT.md) records failed overall quality acceptance and passed speed/correctness checks; this is an experimental package.

## Load the config

Select the package before importing `particlegan` in a fresh Python process. Supply your generator and scalar critic on the same device, then set the structural dimensions for your problem. The example assumes the generator consumes 128 latent coordinates.

```python
import json
from pathlib import Path
import sys

root = Path("/ml2/hypergan/gan-attempts/feature-cells-config-20260929")
sys.path.insert(0, str(root / "pkg-CB64-RA"))

from particlegan.recipes import Recipe
from particlegan.training import GANTrainer

options = json.loads((root / "configs/overrides-CB64-RA.json").read_text())
options.update(z_dim=128, num_particles=1024, batch_size=128)
recipe = Recipe(**options)
trainer = GANTrainer(recipe, G, D, seed=314159)

# During training, real_batch has the same output shape as G.
stats = trainer.step(real_batch)
repair_stats = trainer.birth_death.diagnostics()
```

The generic JSON retains E22's structural defaults of 12 particles and latent dimension 4. Change these to match your model. A critic needs a learned `nn.Linear` scalar score head; raw-input score heads alone are refused by the inherited feature extractor.

## Added settings

| Setting | Value | Meaning |
| --- | --- | --- |
| `birth_death_backend` | `feature_cells` | Select the new reaction and sampling kernel |
| `birth_death_cells` | 64 | Maximum cells fitted from the real reference half |
| `birth_death_metric_rank` | 8 | Maximum projection rank, capped by active features and reference degrees of freedom |
| `birth_death_chunk` | 256 | Maximum rows in feature and distance blocks |
| `birth_death_parent_policy` | `real_anchor` | Search real cell representatives and at most 256 eligible parents per repair |

All other JSON options are copied from the reference E22 config. Ordinary reaction and isolation are both enabled. The new backend uses exact conditional cell-count tests for ordinary moves and calibrated support scores for isolation. It rebuilds the metric for each FIFO turnover.

Training, sampling, fake-pool generation, and row copies share latent Gaussian jitter with standard deviation 0.025 and a norm cap of 0.05. This kernel replaces DV12's adaptive bandwidth and nearest-table clipping for this backend. Other DV12 training controls remain active.

## Checkpoints and scope

Use the trainer's normal `state_dict()` and `load_state_dict()` methods. Checkpoints record backend settings, private randomness, the real FIFO, and row state. Derived feature caches are rebuilt after loading. Loading a checkpoint from another backend or incompatible settings is refused.

The legacy `knn` backend remains the package default. The feature-cell backend retains the raw-real FIFO, whose memory grows with particle count and output size. A bounded parent search also needs existing supported parents in accessible cells; it cannot create an absent mode by itself.
