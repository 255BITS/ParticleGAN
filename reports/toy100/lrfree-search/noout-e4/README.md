# E4 row-gradient evidence and anchored release (September 28–29)

E4 is the first **3/3 frozen native100** live result in this search whose *new
controller* uses no comparison of generated samples with real samples. It
passes grid100, rotated100, and staggered100 at both 7,000 and 14,000 updates
with the original frozen host, QR initialization, noisy scoring, and zero stream
deviations. This is **diagnostic evidence, not a compliant solution** under
`/ml2/hypergan/lrfree-20260926/REQUIREMENTS.md`: the new
gradient-history window was selected after examining a native-task probe, the
hold reuses the FDR level as a separate threshold, and the higher-LR robustness
arm fails all three native tasks. The inherited base ledger also remains open.

The word *new* matters: the inherited paired birth/death operation estimates
model versus real density for mass transfer. It is an output-versus-data
comparison permitted by the project's reaction sink rule. Ordinary GAN losses
also use generated and real batches. E4's **added** row gate and release rule
do not inspect output clouds, target centres, evaluation samples, or gate metrics.

## How it works

E4 builds on `pkg-seqC`. Each table row keeps an exponentially weighted mean
and variance of gradients observed when that row is drawn for a generator
update. A Hotelling-style test followed by Benjamini–Hochberg at `Q=.05`
flags rows with persistent directional gradients. Flagged rows keep the full
table step when the bulk table rate is lowered; they are excluded from the
bulk stationarity vote. E4 also defers a bulk rate descent while the flagged
fraction exceeds `Q`. Separately, its `anchor` rule accepts a rate increase
only when drift evidence appears at least twice the evidence scale of the last
stationarity verdict. The two changes address drifting rows and premature
rate release, respectively. One-factor ablations in the source research
`/ml2/hypergan/gan-attempts/noout-20260928/RESULTS.md` support both. Exclusion
can be removed on the discriminating rotated 7k run; removing the hold changes
the trajectory and misses grid at 7k and rotated at 14k.

## Frozen native verdicts

Each cell is the live frozen verdict. The parentheses give passing observations
and the longest final streak. All 7k and 14k runs use the original
`screen.py`, SHA-256 `ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c`.

| Run | grid100 | rotated100 | staggered100 |
|---|---|---|---|
| E4, 7k | PASS (23/34, streak 22) | PASS (14/34, streak 14) | PASS (18/34, streak 18) |
| E4, 14k | PASS (50/62, streak 11) | PASS (42/62, streak 42) | PASS (46/62, streak 46) |
| E4, LR ×0.75 at 7k | PASS | PASS | PASS |
| E4, LR ×1.33 at 7k | FAIL | FAIL | FAIL |
| E4 without hold, 7k | FAIL (grid centre .2009σ at step 6250) | PASS | PASS |
| E4 without hold, 14k | PASS | FAIL (late centre and precision) | PASS |

The 7k final-centre RMS values are `.1867σ`, `.1531σ`, and `.1772σ` for grid,
rotated, and staggered. Their 100k holdout centre values are `.1652σ`,
`.1391σ`, and `.1658σ`. Rotated holdout precision is `.9722`, which passes the
frozen `.97` limit but misses the project's `.973` design margin. Exact
headers, results, and noisy verdicts are in [native-receipts](native-receipts/).

## Reproducibility and status

The archived [source](source/) has package digest
`f69349eeda9679c6db04711be9eeebe9b42ed994bc544dd6df585be61e0f2188`.
Its base `pkg-seqC` digest is
`70e7b5f507a2515c879f70738a47921b5bc9ab5427a86f2be87b57f513605524`;
the [patch](source/E4-vs-seqC.patch) digest is
`9f483b628de3552ab859f97be92ce5e2aa9585ffcee03056d2348acc0e3bd2fc`.
The archive package reproduces the source package digest exactly. Run the
frozen host with `--package-root` set to `source/`, `--overrides` set to
`source/overrides.json`, and candidate options `{"eval_output_noise": true}`.
For a 14k continuation add `"native_steps": 14000`.

The CPU checks reproduced disabled-flag parity with `pkg-seqC`, checkpoint
replay, and step-offset invariance. A decoy real-reservoir stream left the E4
training digest identical while 4,323 hot-row steps executed. The independent
review's 30 focused code checks passed, including exercised hot-row, hold,
exclusion, and anchor branches. Those checks also found anti-conservative
row-test p-values under anisotropic and autocorrelated nulls. The first CPU
toy's flag-on digest equalled the base because it held the table for most of
the run, so the focused branch checks are the meaningful integration evidence.

The full follow-up task suite and leaderboard are recorded in
[suite-leaderboard.md](suite-leaderboard.md): **12 PASS, 2 FAIL, 8 ERROR** on
the 22-task preset; supplemental ring_shift and stationary both PASS. The two
failures are `img_intensity2` and `img_bars4`; all eight custom hosts refuse
the new trainer hooks at parity before scoring. E4 must not be promoted solely
from its native 3/3: the class-A2 constant provenance and robustness findings
are blocking under the current requirements.

An [oracle feasibility check](oracle-feasibility.md) rejects an initially
proposed disjoint-block, empirical-tail replacement before GPU training:
under strong gradient correlation, even a known-direction test needs an
effect much larger than a weak persistent row gradient within 128 touches.
