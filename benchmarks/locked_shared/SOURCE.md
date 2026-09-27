# Source and reproduction

These CPU experiments are extracted from
[HyperGAN/conceptmod at 5571213f5e8e129cfda45c785c3f30aad9c1d8c9](https://github.com/HyperGAN/conceptmod/tree/5571213f5e8e129cfda45c785c3f30aad9c1d8c9/conceptmod/toys).
The MIT notice is retained in [LICENSE](LICENSE).

| Local experiment | Original source | Kept |
| --- | --- | --- |
| `two_pole.py` | `leaderboard_honesty.py` | Stored host weights, live/stranger training, 80 steps, travel and slope thresholds |
| `trajectory.py` | `shared_trajectory.py` | Arcs, models, full loss, 400 steps, shared/stranger/nearest pairing, identity MSE threshold |
| `mode_hold.py` + `mlp.py` | `mode_hold.py` + `mlp.py` | Ring data, models, 1,200 steps, coverage/HQ metrics and verdict. Problem only: optimizers, LR schedule, loss, penalty, prior, noise and EMA come from the shipped recipe through `benchmarks/toy_runner.py`, so its reference parity no longer holds |
| `hosts/residual_student.py` | `residual_student.py` | Conditional residual head, landing objective and all three measured landing/identity bounds |
| `hosts/unipolar.py` | `unipolar.py` | Positive-pole training, neutral hold, leakage and coverage |
| `hosts/ae_gan_hold.py` | `ae_gan_hold.py` | AE encoder/decoder training, reconstruction and unconditional hold |
| `hosts/cover_leftover.py` | `cover_leftover.py` | Guarded teacher, particles, live/EMA residual geometry and all six bounds |
| `hosts/unused_token_hold.py` | `unused_token_hold.py` | Slot student, concept training and unused-slot hold |
| `hosts/mid_scale_identity.py` | `mid_scale_identity.py` | Four-scale training, polarity/magnitude and identity checks |

The additional hosts remove formulation refusals and unused config gate code;
their models, random draws, budgets and numerical training operations remain.
`host_reference.py` pins the original hashes and checks default numerical parity.
The expanded [baseline protocol](BASELINE.md) defines the full scope and explains
which application checks remain in the reference repository.

Changes: remove configuration equality/refusal gates, reporting requirements,
source-local logging and caches; inject `make_gan_loss()` and `make_b_cap()`
from ParticleGAN PR #36 into the default training paths. Alternatives run
through explicit constructor hooks. Tests check measured outcomes and compare
complete training results with the original constructors. No test requires
an alternative to fail merely because its settings differ.

The host supplies cover, particle L2, VICReg, latent width, model architecture
and optimizer loop. PR #36 supplies the loss and penalty builders, not those
host training loops. In particular, trajectory and ring keep the original
4-dimensional latent and VICReg 0.05; ring has no cover training term.

## Run locally

From the repository root, with CPU PyTorch and pytest installed:

```bash
python -m benchmarks.locked_shared --output runs/locked_shared
python -m pytest tests/test_locked_shared_behavior.py tests/test_locked_shared.py tests/test_api_primitives.py -q
```

The command trains one row per host toy (two-pole, trajectory, ring), logs a
`start`/`done` line per toy, then writes a readable `README.md` and full
precision `results.json`. It exits nonzero if a behavioral target fails.
Formulation and pairing arms (thinned cap, stranger pairings, cap-off,
vanilla, FM-on) are no longer part of the suite.

Rel-1e-6 parity with the unchanged conceptmod source (`reference.py`) and
PyPI 0.5.0 primitive hashes was verified for the original extraction and is a
frozen record in [reports/locked_shared](../../reports/locked_shared/README.md).
It is not rerun: hosts migrated to `benchmarks.toy_runner` take optimizers,
loss, penalty, noise and EMA from their recipe, so they no longer replay the
original loops. `reference.py`, `host_reference.py` and `suite_reference.py`
are kept only as the scripts that produced those records.

This is a selected behavioral leaderboard, not the original suite's mix of
configuration checks, geometry checks and DSL claims. There is no score for
untested variant/toy combinations and no seed sweep. See the checked-in
[results](../../reports/locked_shared/README.md) for the actual verdicts.

## Full reference row and formulation comparisons

The optional `suite_reference` audit calls only the original `locked_shared`
row and excludes cover-posture columns. It needs the reference project's
dependencies, including PEFT 0.21 for its LoRA-path toy. Its original scorer
results are recorded separately from the extracted behavioral leaderboard:

```bash
python -m benchmarks.locked_shared.suite_reference --reference /path/to/conceptmod
```

The recorded [comparison report](../../reports/locked_shared/comparison.md)
tested existing GAN losses, penalties and host regularization choices on the
same seed and budgets. The scripts that swept those formulations
(`investigate`, `summarize`, `base_recipe`, `default_selection`) were removed:
toys now define only their problem and train on their recipe.

Optional diagnostics record live versus EMA ring quality, learning curves,
trajectory nearest-target assignments and critic slopes. They do not alter
the default losses or reference comparison. The ring host accepts Ra as well
as Rp real logits for the comparison, and an optional stock `Recipe` can
supply its prior, optimizer groups and learning-rate schedule. Production
`particlegan` modules are unchanged.
