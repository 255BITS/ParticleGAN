# BCAP Tier 1 numerical task audit

**Retain the Gaussian and ring gates for the first repair search.** BCAP's
retained failures are substantial distribution errors, rather than demonstrated
scorer mistakes. **The selected K3P ring run passes**; its Gaussian run fails
the terminal stability requirement despite a passing endpoint. Shared failure
is therefore not evidence that both acquisition questions are unsolvable.

This audit adds **zero training updates and zero model sampling draws**. It
recomputes all 96 retained BCAP/K3P observations, checks byte-exact archived
receipts and sample tensors, validates the original oracle/destructive controls,
and runs isolated analytical sampling controls. The
[compact audit](task-audit.json) records exact identities and uncertainty;
[the current leaderboard](../technique-inventory.md) remains the sole ranking.
Original qualification results are unchanged.

## Actual failures and comparable evidence

Both formulations use the same task, prior, initialization, named RNG bindings,
budget and clean/live sampling law within each question. Their **recipes differ**:
BCAP uses LR .00425, D multiplier 2, prior multiplier 4 and coefficient .5;
K3P uses LR .006375, D multiplier 1, prior multiplier 1 and coefficient 1.
This comparison cannot isolate the critic formulation as a causal factor.

| Retained run | Original verdict | Numerical explanation |
| --- | --- | --- |
| BCAP Gaussian, `d0638ad5ce5a47e5b2fcb00b369768b2` | FAIL | Final mean error .4170 sigma and KS .1750; width ratio 1.0826 passes. All five terminal KS checks fail. |
| K3P Gaussian, `7ccd8797a0ab4592b312a86354518228` | FAIL | Final KS .0260 passes, but checks 834/875/917 have KS .07385/.07751/.05297; passing suffix is two, not five. |
| BCAP ring, `d8185ca486b54af79ef422395eac8065` | FAIL | All 16 modes, mass TV .1221 and precision .9385 pass; full component covariance error 4.4570 fails .85, and minimum eigen ratio .1345 fails .15. |
| K3P ring, `d6d907c833524c1ea23374d90c548cf6` | PASS | Five terminal checks at 334–400 pass; final full covariance error .6216 and minimum eigen ratio .3238 pass. |

The exact original sources are preserved in the
[Tier 1 completion artifact inventory](../tier1-completion/artifact-inventory.json).
The archive SHA-256 is
`2041cfe2e9975b533089d4c74b0970164235c0d0b35af46a3ebbd7b7168b9c66`.
The audit verifies archive members against its manifest, result certificates,
raw grading hashes, effective overrides, complete update counts, finite-state
guards and all 24 checks per run. Maximum gated-metric recomputation error is
recorded per run; all comparisons satisfy absolute/relative tolerance 2e-6.

## Gaussian: drift and finite-draw uncertainty are distinct

BCAP's last five KS distances are .11148, .14178, .16525, .17362 and .17496.
Distribution-free DKW bounds, with a union bound over those five states, give a
99% simultaneous evaluation uncertainty radius **.02904** at 4,096 iid samples.
Every terminal KS lower bound exceeds .05; the endpoint interval is
**[.14592, .20400]**. This rejects a sampling-noise explanation for the BCAP CDF
failure. Its mean error also grows to .417 sigma; endpoint mean standard error
is only .0169 sigma.

For diagnosis only, recentering BCAP's endpoint reduces KS from .17496 to
**.03916**; matching both saved moments gives .03836. The same diagnostic at
K3P's first two failing terminal checks reduces KS to .03507/.02844. Location
drift therefore explains much of the observed mismatch, but these transformed
arrays neither regrade the original output nor demonstrate a trainer repair.

K3P's first two terminal KS intervals straddle .05. The audit cannot determine
from one saved draw per state whether its population CDF passes. Its observed
.1407/.1703-sigma location errors are also meaningful: a perfect Gaussian shifted
by .1703 sigma already has population KS about .0679. K3P's shared failure is
not proof of an erroneous CDF criterion.

The all-five empirical rule nevertheless rejects some *population-acceptable*
laws too often. In **400 independent five-check panels per analytical law**:

| Fixed analytical law | Population KS | Individual check rejection | Five-check panel rejection |
| --- | ---: | ---: | ---: |
| Exact target Gaussian | 0 | 0% | 0% |
| Shift .075 sigma | .02991 | 2.00% | 9.50% |
| Shift .10 sigma | .03988 | 25.55% | 77.25% |
| Shift .15 sigma | .05979 | 97.60% | 100% |
| Shift .25 sigma | .09948 | 100% | 100% |

All these draws use one isolated diagnostic RNG, with no trained model or
training-seed variation. Binomial uncertainty is recorded in the JSON. For the
exact oracle, the DKW five-check false-rejection upper bound is only
**1.28e-8**; near-margin rejection is a different issue. These controls measure
the empirical protocol's tolerance margin, not training reliability or Tier 1
calibration.

**Proposal:** if the question is intended to accept population KS up to .05,
register a later variant with a larger evaluation sample count, preserving the
bound and five terminal states. At 32,768 samples, the same 99% simultaneous
DKW radius is .01027 rather than .02904. Validate that variant's controls and
cost before adopting it uniformly across techniques. Do not relax KS or use
moment-recentered samples to make the present BCAP run pass.

## Ring: tails dominate, while some cores remain thin

In BCAP's final ring draw, **5.737%** of samples lie beyond four target sigmas,
and account for **86.87%** of centered within-component covariance energy.
Mean radial variance is **6.13 times** target, versus tangential variance
**1.12 times** target. The full covariance failure primarily reflects radial
tails, rather than lack of modes or a global ring-radius error.

The four-sigma core covariance error is still .7124, and core minimum eigen
ratio .1296. Thus removing tails would not establish healthy local shape.
Components have 97–481 assigned samples; imbalance increases uncertainty in
the least populated components. Exploratory iid bootstrap intervals from 300
resamples are **[3.79, 5.17]** for full covariance error and **[.704, .731]**
for core covariance error. The gated full minimum eigen interval
**[.113, .154]** straddles .15; the core interval **[.095, .134]** supports the
thin-core diagnosis. Bootstrap intervals are conditional on retained samples,
not evidence of repeat-training reliability.

K3P's final full covariance interval **[.569, .699]** remains below .85,
with all component minimum eigen ratios above .15. Its successful trajectory
demonstrates that the current host, 400-update budget and gates are achievable
by at least one existing recipe. It does not prove BCAP needs no larger budget.

Two added controls limit tempting scorer revisions:

- Four same-covariance atoms in each cluster pass the current ring gate.
  This gate establishes acquisition, balance and local second-moment spread;
  it does not establish Gaussian density fidelity. The existing scalar CDF
  controls correctly reject same-moment discrete and uniform impostors.
- A balanced Gaussian core with **1.95% nine-sigma outliers** fails the full
  covariance bound (1.0712), but passes a proposed replacement using core
  covariance plus per-component spill <=.05. Replacing the current full
  covariance metric would admit a material tail failure. Keep full covariance
  for this repair; use core/tail metrics as diagnostics.

**Proposals:** slow prior transport and strengthen the existing penalty before
altering the distribution question. Test extra settling separately while
preserving the original schedule horizon and prefix checks. BCAP's late curve
is not steadily improving in all metrics: full error is still 4.46 at update
400 and cores thin late. These saved outputs do not support an automatic
unchanged continuation or a guarantee that more training alone fixes it.

## Two-pole budget, validation and reproduction

The fixed 80-update two-pole gate measures direct-particle travel >=.3 and
critic slope <=1; it does not require complete two-mode acquisition. Its
direct-coordinate group consumes **base Recipe.lr**, while `prior_lr_mult`
only scales sampled latent tables. Raising that multiplier therefore cannot
offset conservative global LR on this host. The direct response can raise
its scheduled LR by at most twofold.

The existing
[800-update horizon diagnostic](../k3p-two-pole-horizon-v1/README.md)
already separates slow motion from local force balance: the low-rate,
coefficient-170 word recipe remains below .112 travel under either horizon,
while the movement control passes both. That comparison changes several recipe
factors and does not establish that 80 updates is universally wrong. Retain the
original gate for the first search; if slower LR fixes Gaussian/ring but fails
travel, inspect actual forces and trajectory before registering a task-budget
revision. The JSON retains the original horizon-study source digest and arms.

All 8 original Gaussian and 9 original ring oracle/destructive controls behave
as declared. Global Torch RNG is unchanged. No training code, task declaration,
selection or leaderboard is modified by this audit. Existing actual-training
illustrations remain available for
[K3P Gaussian](../tier1-completion/media/k3p/7ccd8797a0ab4592b312a86354518228/gaussian1d_acquisition.gif)
and [K3P ring](../tier1-completion/media/k3p/d6d907c833524c1ea23374d90c548cf6/ring16_acquisition.gif);
the numerical evidence drives conclusions.

Reproduce with the declared archive present, or provide its byte-exact local
copy using `--archive`. Write progress outside Git for easy tailing:

```sh
mkdir -p runs/forge/bcap-tier1-repair
PYTHONPATH=. python -u reports/forge/bcap-tier1-repair/audit_tasks.py \
  --output /tmp/bcap-task-audit.json \
  > runs/forge/bcap-tier1-repair/task-audit.log 2>&1
tail -F runs/forge/bcap-tier1-repair/task-audit.log
cmp /tmp/bcap-task-audit.json reports/forge/bcap-tier1-repair/task-audit.json
```

The audit CLI exits on any receipt/hash/recomputation/control mismatch. The
committed JSON contains compact metrics and provenance only; raw outputs,
checkpoints and per-update traces remain in the existing artifact archive.
