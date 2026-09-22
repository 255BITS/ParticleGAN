# Source and reproduction

These CPU experiments are extracted from
[HyperGAN/conceptmod at 5571213f5e8e129cfda45c785c3f30aad9c1d8c9](https://github.com/HyperGAN/conceptmod/tree/5571213f5e8e129cfda45c785c3f30aad9c1d8c9/conceptmod/toys).
The MIT notice is retained in [LICENSE](LICENSE).

| Local experiment | Original source | Kept |
| --- | --- | --- |
| `two_pole.py` | `leaderboard_honesty.py` | Stored host weights, live/stranger training, 80 steps, travel and slope thresholds |
| `trajectory.py` | `shared_trajectory.py` | Arcs, models, full loss, 400 steps, shared/stranger/nearest pairing, identity MSE threshold |
| `mode_hold.py` + `mlp.py` | `mode_hold.py` + `mlp.py` | Ring data, models, full loss, EMA, 1,200 steps, coverage/HQ thresholds |
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
python -m benchmarks.locked_shared --output reports/locked_shared
python -m pytest tests/test_locked_shared_behavior.py tests/test_locked_shared.py tests/test_api_primitives.py -q
```

The command logs each arm before and after training, then writes a readable
`README.md` and full precision `results.json`. It exits nonzero if a default
behavioral target fails or a requested reference comparison differs. A
benchmark failure and a parity match can both be true. The pytest checks for
builder equivalence do not assert that an existing failing reference passes.

To verify extraction against the unchanged original source as well:

```bash
git clone https://github.com/HyperGAN/conceptmod.git /tmp/conceptmod-reference
git -C /tmp/conceptmod-reference checkout 5571213f5e8e129cfda45c785c3f30aad9c1d8c9
python -m pip download --no-deps particlegan==0.5.0 --dest /tmp/particlegan-reference
python -m benchmarks.locked_shared \
  --reference /tmp/conceptmod-reference \
  --reference-wheel /tmp/particlegan-reference/particlegan-0.5.0-py3-none-any.whl \
  --output reports/locked_shared
```

No conceptmod installation or diffusion dependencies are needed. The optional
reference runner verifies SHA-256 hashes of the four original toy files and
loads them unchanged, bypassing the package's eager backend imports. Its
original constructors use the same ParticleGAN primitives as the extracted
loops. The optional wheel check independently verifies that those primitive
sources also match PyPI 0.5.0. Parity covers all ten rows and all numeric
metrics, including negatives, at relative tolerance 1e-6 and absolute 1e-7.

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

The [comparison report](../../reports/locked_shared/comparison.md) tests
existing GAN losses, penalties and host regularization choices on the same
seed and budgets. It includes separate stock-recipe ring runs with 20,000
particles and explicitly labels the longer 7,000-step budget. Commands and
raw measurements are linked there. `investigate --resume` reuses completed
ring runs and fills missing diagnostics; all training variants can also be
reproduced from scratch.

Optional diagnostics record live versus EMA ring quality, learning curves,
trajectory nearest-target assignments and critic slopes. They do not alter
the default losses or reference comparison. The ring host accepts Ra as well
as Rp real logits for the comparison, and an optional stock `Recipe` can
supply its prior, optimizer groups and learning-rate schedule. Production
`particlegan` modules are unchanged.
