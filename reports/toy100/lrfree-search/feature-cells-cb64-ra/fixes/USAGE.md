# Corrected feature-cell experiment

The corrected candidates are experimental packages. Use the package and config
with the same suffix in a fresh Python process. E22 remains the reference.
No corrected package is recommended for the current target until it passes
both the unchanged learned toy and canonical grid gates and the required
state, replay and portability checks.

```python
import json
from pathlib import Path
import sys

root = Path("/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929")
variant = "CB64-RA8"  # experimental; toy passes, full grid fails
sys.path.insert(0, str(root / f"pkg-{variant}"))

from particlegan.recipes import Recipe
from particlegan.training import GANTrainer

options = json.loads((root / "configs" / f"overrides-{variant}.json").read_text())
options.update(z_dim=128, num_particles=1024, batch_size=128)
trainer = GANTrainer(Recipe(**options), G, D, seed=314159)
stats = trainer.step(real_batch)
```

Supply a generator consuming the specified latent width and a compatible critic
with a learned linear scalar head. Models and real batches must use the same
device. The example assumes `G`, `D` and `real_batch` have already been defined.
The archived config retains generic E22 structural defaults of N=12 and z=4;
set these to match the actual model.

RA2 through RA6 have identical option values. Their package
code implements different explicitly recorded corrections; swapping the JSON
alone does not install a correction. Populations below the finite count-test
resolution boundary N=800 use the established reference backend. The selected
and actual backend are recorded in diagnostics and checkpoints.

RA2 introduces exact batched count tests, real-reference mass targets, unique
parents and bounded local latent jitter. RA3 adds broad-flag count recovery and
a bounded copy-lineage graph. RA4 adds support-aware count
categories, exact planner/cache batching and the canonical indexed-generation
API. See the frozen composition and contract receipts for the final content.

RA5 adds separate live/EMA copy bounds, up to four paired novel latent births
per reaction, and a population coverage requirement for stationarity evidence.
The births use observed real examples and learned critic features, with at
most four local linearizations per model and rank at most eight. They consume
the existing shared action budget and count-test family. RA6 fixes JSON
serialization of birth diagnostics; the numerical operations are unchanged
from RA5. RA5's failed GPU logging attempt is retained and has no quality
verdict. RA6 uses backend schema 6 and trainer schema 5.

RA7 retains those schemas and mechanisms. Its integer group-count reduction
has exact CPU/CUDA planner evidence. Its config changes only `lr` to
0.0010625, `prior_lr_mult` to 8 and `d_lr_mult` to 4: the G network and learned
sigma base rates quarter while prior/D bases remain 0.0085/0.00425. Future
stationarity decisions can differ, so applied rates are still state dependent.
The noise formula, floor and serving rules are unchanged.

RA8 keeps that config and training law. It adds an empirical check after each
reaction for paired averaged serving: the averaged anchors must remain in the
current learned support chart and agree with corresponding live anchors on
real-only support groups for at least 95% of the population. Eligibility expires
after one real FIFO turnover. Generator and critic weights can move between
checks, so this is a measured geometry check with bounded age. Emitted samples
still use the learned noise law and must pass the original quality gates.
RA8 uses backend schema 7 and trainer schema 5. Backend schema 6 checkpoints
are rejected before loading model state. The serving decision survives
continuation while derived caches are rebuilt.

Use `trainer.state_dict()` and `trainer.load_state_dict()` for continuation.
Backend schemas/settings and serialized lineage topology are checked before
loading; incompatible candidate checkpoints are rejected. Derived feature and
sorted-coordinate caches are rebuilt. Exact CUDA checkpoint replay is tested
separately from quality.

The raw-real FIFO still grows with N and output size. Novel births also require
bounded generator/critic Jacobians and small singular value decompositions.
The integer count reduction has exact CPU/CUDA evidence; its saved-plan
microbenchmarks do not establish end-to-end scaling laws. Fixed cell count/rank and
bounded neighbor candidates can lose statistical power or miss geometry as
problems become more complex. The package has not established universal
support detection, distributed training, or isolated GPU scaling laws.
RA8 also queries the full averaged population once per reaction; this adds
generator/critic work that grows with population and model size.

For the repository archive, replace `root` with the `fixes/` directory.
Validation runners retain the original local raw-input paths in their frozen
receipts. Raw tensors and datasets are not included in Git.
