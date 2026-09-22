# Behavioral baseline protocol

The [leaderboard](../../reports/behavioral_baseline/README.md) is the common
comparison target for configuration searches and new implementations. Overall
PASS requires every numerical bound on **final live weights** and every shared
behavioral/integration check. EMA results are reported separately. No selection
of the best checkpoint, seed sweep, config-identity gate, or missing-toy exemption
contributes to the candidate ranking.

Regression PASS is the original minimum bar, including **at least 7/8** ring
modes. Default selection additionally needs the actual coverage, sample quality,
mode balance and stability shown on the main leaderboard. A perfect final HQ
score does not penalize an entirely missing cluster. Keep the original 29
regression bounds fixed so new measurements cannot silently redefine PASS.

The ring diagnostics now enumerate every learned particle and record live
checkpoints every 200 steps, plus every 50 steps in the final 200 updates.
They preserve the training RNG and final-step selection. The leaderboard shows
the worst observed tail coverage/HQ and how many of its five tail observations
have all eight modes with HQ at least 90%. These are observations, not guarantees
about intervening updates. Effective modes is the entropy-based balance measure
among HQ samples; exact support counts are also stored. With 12 equally weighted
particles, the most even allocation across eight modes is four modes with one
particle and four with two: its exact effective-mode ceiling is about 7.56.
The 4,096-sample estimate fluctuates around that value.

For the expanded comparison, equal regression pass counts are ordered by live
ring coverage, then HQ, then effective modes. This makes quality improvements
visible among equally passing configs while preserving all original bounds.
The main leaderboard also displays the matched stock-recipe ring comparison
separately. Its recorded source fingerprint is retained; those larger-budget
rows cannot earn passes in the small-host suite. The optional `--stock-reference`
argument selects that recorded comparison, defaulting to the checked-in
`reports/behavioral_baseline/stock_ring.json` when present.

See [default-selection analysis](../../reports/behavioral_baseline/default_selection.md)
for the missing-mode diagnosis and matched penalty comparison using the stock
recipe's capacity, optimizer and schedule on the ring host. The main nine-toy
board remains the same-budget 12-particle comparison. Neither board alone is a
replacement for ParticleGAN's existing 100-Gaussian/default-recipe benchmarks.

## What is included

This inventory accounts for every column in the pinned conceptmod suite.

| Original suite column | Baseline treatment | Evidence |
| --- | --- | --- |
| leaderboard_honesty | Train each candidate (`two_pole`) | Travel and critic gradient median |
| shared_trajectory | Train each candidate (`trajectory`) | Identity MSE |
| residual_student | Train each candidate | Identity MSE, every own-pad landing, zero wrong-pad landings |
| unipolar | Train each candidate | Positive-pole coverage, off-caption leakage, neutral hold |
| ae_gan_hold | Train each candidate | Reconstruction MSE, unconditional anchor distance |
| cover_leftover | Train each candidate | Direction/content retained, leakage, both pole errors, even residual |
| unused_token_hold | Train each candidate | Unused-slot hold and concept movement |
| mid_scale_identity | Train each candidate | Direction and magnitude at both poles; identity at 0 and 0.5 |
| mode_hold | Train each candidate | Ring coverage ≥7/8 and HQ ≥90% |
| orbit_hold | Shared numerical check | Closed-loop direction, speed, radius |
| erase_keep_backend | Shared backend check | Measured host deltas, teacher leakage, coverage, pole geometry |
| late_collapse | Shared selector check | Validation selector avoids the synthetic late collapse |
| keep_critic | Shared numerical check | Frozen host weights do not move during its training loop |
| lm_target | Shared training check | Trajectory expression gain, structure hold, zero-scale identity |
| field_lift | Shared geometry check | Plane and tilted-frame geometry, including lyric leakage |
| path_suffix_lora | Shared integration check | Measured gradients and coverage at intended LoRA targets |
| dsl_macro_expand | Shared parser check | Correct expansion and rejection of seven incorrect expansions |
| dsl_phrase_jobs | Shared application check | All six documented phrases train and satisfy their geometry/semantics |
| dsl_game_geometry | Shared application check | Four phrase geometries after training; GAN stamp comparison removed |
| locked_shared_floor | Excluded | Formulation/config identity |
| particle_posture | Excluded | Posture check, per requested scope |
| posture_demo, posture_music (older suite) | Excluded | Config identity |

The nine trainable numerical hosts are local to ParticleGAN and need only
PyTorch. The ten shared checks run against the pinned conceptmod application
because their backend, DSL and LoRA behavior belongs to that application. They
are evaluated once, independently of the candidate, and earn **no ranking
points**. We do not label a reference app's PASS as evidence that an alternative
GAN was trained inside its DSL/backend. The full baseline is INCOMPLETE if these
checks are missing or error. A measured failure makes it FAIL.

Each candidate has 29 required numerical bounds. Ring `effective_modes`,
trajectory set coverage, and the unipolar negative-scale canary remain recorded
diagnostics: the source provides no acceptance threshold for them. We preserve
the source thresholds rather than invent thresholds after seeing a result.
The source's combined formulation verdicts are ignored.

## Candidate configuration

Candidates are JSON objects with a unique `name` and optional fields below.
Unspecified fields use the locked-host defaults, regardless of the name. The
checked-in [configs](../../reports/behavioral_baseline/configs.json) contain the
fully resolved values. Unknown fields are rejected to catch misspelled settings.

| Field | Default | Applied to |
| --- | ---: | --- |
| loss_type | logistic | All nine candidate training hosts |
| gan_mode | rp | All nine candidate training hosts |
| reg_arm | b_cap | All nine candidate training hosts |
| reg_coeff | 1.0 | All nine candidate training hosts |
| reg_kappa | 1.0 | All nine candidate training hosts; only affects penalties that use kappa |
| particle_l2 | 0.02 | Two-pole, trajectory, residual student, AE-GAN, cover/leftover, ring |
| vicreg_weight | 0.05 | Trajectory, residual student, cover/leftover, ring |
| cover_weight | 1.5 | Trajectory, residual student, AE-GAN, cover/leftover, mid-scale |
| lr_multiplier | 1.0 | Both host optimizer learning rates in all nine candidate toys |

Unipolar and unused-token hosts have no particle cloud. Two-pole has no VICReg
or cover training loss. Ring has no cover training loss. The AE host retains its
original reconstruction objective and encoder. Cover/leftover retains its
VICReg target standard deviation 0.05; trajectory/residual/ring retain their
original standard-deviation target. Fixed data, pairing, architecture, capacity,
hold/reconstruction objectives, optimizer betas, schedules, EMA decay, evaluation
samples and step budgets belong to the host protocol. A candidate cannot gain
extra compute or per-toy tuning through this JSON schema.

The `locked_shared` baseline uses the same GAN core as stock `Recipe('gan')`.
It is the small-host baseline, not the stock recipe's 20,000-particle/7,000-step
training run. That separate comparison is in the
[earlier report](../../reports/locked_shared/comparison.md).

## Running and extending the baseline

```bash
git clone https://github.com/HyperGAN/conceptmod.git /tmp/conceptmod-reference
git -C /tmp/conceptmod-reference checkout 5571213f5e8e129cfda45c785c3f30aad9c1d8c9
# Use an environment with the reference app's dependencies. The LoRA check needs
# PEFT >=0.21; recorded dependencies are in results.json:shared_environment.
python -m benchmarks.locked_shared.baseline --reference /tmp/conceptmod-reference \
  > /tmp/behavioral-baseline.log 2>&1
tail -f /tmp/behavioral-baseline.log

# New output folder for another candidate set. JSON is a list of candidate objects.
python -m benchmarks.locked_shared.baseline --configs candidates.json \
  --reference /tmp/conceptmod-reference --output reports/my_approach

# Continue an interrupted run with exactly the same config/source/runtime.
python -m benchmarks.locked_shared.baseline --configs candidates.json \
  --reference /tmp/conceptmod-reference --output reports/my_approach --resume
```

Each completed toy is saved atomically before the next toy begins. Failures are
reported without stopping other candidates. Exit 0 means at least one candidate
fully passes; exit 1 means there is no full PASS. `--resume` rejects changed
configs, source, thresholds, budgets or runtime. It retries errored toys. It
does not accept old results from a different experiment under the same label.

New formulations can extend the loss/penalty factory in `Candidate` while using
the same `run_toy` hosts and metric definitions. Keep VERSION, thresholds, seed,
budgets, data, evaluation and hosts fixed for a comparable experiment; record
the changed source fingerprint. Change the protocol version if evaluation or
host training changes. Never select thresholds or checkpoints to rescue an arm.
For later search agents, assign independent candidate files and output folders;
do not edit shared host modules or reuse one result directory concurrently.

## Extraction verification

Six additional hosts were extracted from the same reference commit as the
original three. Their measured default outputs are compared against unchanged,
SHA-256-pinned originals by:

```bash
python -m benchmarks.locked_shared.host_reference --reference /tmp/conceptmod-reference
python -m pytest tests/test_behavioral_baseline.py tests/test_locked_shared_behavior.py \
  tests/test_locked_shared.py tests/test_api_primitives.py -q
```

The [parity artifact](../../reports/behavioral_baseline/extraction_parity.json)
records reference values, extracted values and differences. Cover/leftover's
original final result uses EMA; extraction parity compares that same EMA, while
the new leaderboard additionally captures the live residual before EMA copying.
AE-GAN's source config audit resets and advances the global RNG before training.
The extraction preserves that exact data-stream offset for every candidate,
without retaining the audit's config/penalty comparisons.
The original three-host parity evidence remains in
[results.json](../../reports/locked_shared/results.json).

Source license and provenance: [SOURCE.md](SOURCE.md), [MIT notice](LICENSE).
No production defaults are changed by these experiments.
