# BCAP Tier 1 repair

The bounded round is complete: **no whole BCAP recipe passed all seven current
Tier 1 tasks**. Both searches reached at most **4/7**. The selected incumbent and
its recorded qualification remain unchanged. This is a research result under a
provisional profile, not a default adoption or a 100% claim.

Twelve new whole recipes and two incumbent duration diagnostics produced **86
certified task results: 40 PASS, 46 FAIL**, for **2,687.374 paid seconds** on the
two RTX A6000 GPUs (CPU control costs included). All reservations are released.
The declared campaign ceilings total 31,800 seconds within the initial
36,000-second cap. No seed-only trials, scientific retries or higher tiers ran.

[Final outcomes](results.json), [independent numerical verification](rates-evidence.json)
and [actual-training GIF receipts](media.json) retain every attempted recipe.
The [single current leaderboard](../technique-inventory.md) links this readout;
these new source cohorts do not replace its older selected rows.

## What failed and what changed

The [task audit](task-audit.md) distinguishes failure mechanisms. The incumbent
Gaussian had a 0.417-sigma location error and KS 0.175 at 1,000 updates. Its late
standardized shape was substantially better, motivating slower transport. The
ring acquired all 16 modes, but 5.74% of samples beyond four component standard
deviations carried 86.87% of covariance energy; its cores were also too thin.
Replacing full covariance with a core-only criterion would incorrectly pass a
balanced distribution with rare severe tails. The original numerical thresholds
were retained.

The current selected K3P ring **already passes**. Its Gaussian endpoint passes,
but earlier terminal checks fail. Those outcomes do not establish that both new
tasks have bad criteria: they use different recipes, and endpoint accuracy is
separate from sustained accuracy. Gaussian controls do reveal finite-sample
false rejection near the KS margin, so task calibration remains unfinished.

Tier 1 now asks scheduled recipes to pass an explicit **schedule-contract audit**,
checking independently calculated LR, noise, Adam beta2, penalty coefficient and
per-parameter guard behavior against actual public-trainer updates, restart and
observation cadence. The oracle covers the cosine interior and the guard release
boundary. Passing it makes no clock-free claim. The strict clock-free gate is
retained separately, with its original archived failures and view revisions.

Forge's vector adapter now supports a separately declared execution duration
while retaining the original schedule horizon. Admission and dispatch bind the
exception to the original host, prior, initialization, scorer and sampling law;
forged budget or source changes are rejected. These variants are diagnostics
and confer no ordinary qualification.

## Measured repair tracks

| Track | Declared contrast | Result |
| --- | --- | --- |
| Rate search, 8 recipes | LR .002125/.0010625; prior multiplier .5/1; coefficient 1/2; D multiplier 2, cap 1 | Gaussian 3/8 PASS; word 7/8 PASS; movement and ring 0/8 PASS; best whole recipe 4/7 |
| Incumbent duration, 2 tasks | Gaussian 3,000 updates at original horizon 1,000; ring 1,600 at horizon 400 | Both FAIL; all 24 original observations and scored sample tensors reproduced exactly |
| Adam memory search, 4 recipes | LR .00425; beta2 .9/.99; prior multiplier 1/2; coefficient 1, D multiplier 2, cap 1 | Movement and AE 4/4 PASS; Gaussian, ring and word 0/4 PASS; best whole recipe 4/7 |

All twelve operational schedule audits pass. The searches complete all seven
independent Tier 1 jobs even after a numerical failure; task-specific winners
cannot be assembled into a passing recipe.

[Duration verification](duration-audit.md) shows why simply training this incumbent
longer is insufficient. Gaussian mean error improves to **0.0012 sigma**, but all
five late CDF checks fail, with KS **.066–.125** against .05. Ring covariance falls
from **4.457 to 3.027**, still above .85, and minimum spread remains below .15.
Original checkpoints were not retained, so the exact sample/metric prefix proof
does not establish complete original optimizer/RNG-state parity.

The shorter-memory search improves some final ring covariances (best **1.385**).
Movement recovery is consistent with restoring the global LR; direct-coordinate
Adam betas remain unchanged. The search does not deliver sustained acquisition or joint-word
stability. Several endpoints pass individual checks while the terminal window
fails; final metrics alone cannot substitute for the certified gate.

## Recommendations

Keep the incumbent, the Gaussian/ring bounds and the ordinary budgets for this
PR. Do not extend these exact failed recipes or promote any searched recipe.
The evidence supports two next bounded investigations:

1. Separate optimizer roles through the public recipe API. Test a fixed generator
   beta2 with shorter critic memory, or a distinct direct-coordinate rate, against
   the observed Gaussian stability versus movement/word tradeoff. Use one whole
   recipe across tasks; first inspect saved critic/gradient and update-scale
   diagnostics. Shorter shared moment memory is not a demonstrated root cause.
2. Calibrate the new acquisition tasks. Add declared near-boundary population
   controls and a larger Gaussian evaluation sample count to measure false
   rejection without relaxing the KS bound. Bind every revised observation law
   to a new task identity and preserve old failures. Scorer controls establish
   numerical behavior; they do not calibrate acquisition duration or Tier 1
   placement. A ring revision must retain a tail-sensitive distribution check.

A further training round needs its own finite declaration and prediction. The
remaining allowance is not a reason to repeat unchanged failed experiments.

## Provenance and reproduction

Rates executed source commit `1f0cf74c`, digest `a16f577e680b…`; moments executed
`2cdce7be`, digest `888d8710147b…`. Duration diagnostics executed `a7e4cdf5` with
that second digest. Its source difference enables duration admission; the
independent audit verifies unchanged trainer/scorer files and exact original
sample/metric prefixes. The two search cohorts are reported separately.

[Plans](plans.json), [moment plans](moment-plans.json), the immutable
[first-round readout](first-round.json), [duration plans](diagnostic-plans.json)
and committed recipe cards bind the original gates and full task allowances.
The [publication guide](publication.md) describes independent frozen-source
grading and hash-bound navigation without changing family selection.

Execution is complete; these commands inspect and regenerate receipts without
starting new training:

```sh
.venv/bin/python reports/forge/bcap-tier1-repair/run.py report
.venv/bin/python reports/forge/bcap-tier1-repair/optimizer_search.py report
.venv/bin/python reports/forge/bcap-tier1-repair/publish.py report
tail -F runs/forge/bcap-tier1-repair/queue/events.jsonl
```

The preparers reject already admitted studies. Reproduction sources are
`prepare.py`, `run.py`, `optimizer_search.py`, `audit_tasks.py` and
`audit_duration.py`. [The byte-verified archive](archive.json) retains original
stdout, scored tensors, certificates, queue events and source snapshots at
`/mnt/ml7tb/experiments/ParticleGAN/bcap-tier1-repair-v1/artifacts.tar.gz`.
Bulk artifacts and software test logs stay out of Git.
