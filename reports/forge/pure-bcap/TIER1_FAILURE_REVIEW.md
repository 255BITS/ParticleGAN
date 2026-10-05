# Why BCAP fails its remaining three Tier 1 tests

**Recommendation: continue one bounded, targeted search; retain all three tests
and their numerical requirements.** The selected BCAP recipe is unstable on the
Gaussian, leaves substantial ring tails, and loses an already learned word
solution. These failures are too large to justify threshold relief. Existing
controls and trained alternatives also argue against disabling the tests.
Continue search means a finite comparison of training balance, with a stopping
rule; it does not mean repeating the completed loss/rate grid indefinitely.

This is a read-only review against `origin/develop` at
`0115a92f68dbf9bdcdf9e4f7bfea0fff8606e752`, dated 2026-10-05. Three subagents
independently explored Gaussian, ring and word evidence. No training, new model
sampling, qualification changes or threshold changes were performed.

## Which BCAP and which failures?

The [current leaderboard](../technique-inventory.md) calls the native-Adam family
**BCAP**, ID `bcap-pure`. Its selected whole configuration is
[`5ea5bdbb2d71`](../../../configs/forge/configurations/bcap-pure--5ea5bdbb2d71403dd316e201a51fb2b2b9c1868a8e053156b21b7cecb344d4be.json):
paired relativistic logistic loss, constant global LR `0.00425`, critic multiplier
`1`, prior multiplier `2`, Adam betas `(0, 0.999)`, and fixed BCAP coefficient/cap
`1/1`. It uses no clipping, A2, anchor, EMA serving or additive training noise.

The [BCAP family report](../families/bcap-pure.md) records **3/6 required
discriminator-stability passes**: two-pole, unused-token hold and AE/GAN hold.
Gaussian, ring and word fail. The separate clock diagnostic passes. The displayed
**19/22 across views** counts shared requirements repeatedly; it is not 19
independent successful experiments. Recorded qualification is tier 0.

**BCAP with K3P**, ID `bcap`, is a different trainer. Its word PASS and clock FAIL
cannot replace this family's cells. Its schedule/noise explanations and longer
duration diagnostics cannot establish the cause of native-Adam BCAP's failures.
This report uses the existing leaderboard and adds no second generated board.

## Numerical failure summary

Every acquisition test requires **five consecutive passing terminal checks** out
of 24 clean/live observations. Endpoint success or an earlier good checkpoint
does not satisfy that claim. These are the selected recipe's certified results:

| Test / budget | Endpoint failures | Endpoint bounds that pass | Passing observations / terminal suffix | Decision |
| --- | --- | --- | --- | --- |
| [Gaussian](receipts/80f86665c2924acabb3807936ca9af19.json), 1,000 updates | CDF KS `0.096744 > 0.05`; width ratio `1.261683 > 1.2` | Mean error `0.007826σ ≤ 0.2σ`; finite output; 4,096 samples | `2/24`; `0`, required ≥5 | Continue bounded search |
| [Ring16](receipts/aae235aaaaa64ac2b5e36c18f4af3155.json), 400 updates | HQ `0.822754 < 0.85`; full component covariance error `5.366516 > 0.85` | 16 modes; mass TV `0.099854 ≤ 0.15`; minimum eigen ratio `0.348120 ≥ 0.15`; 4,096 samples | `0/24`; `0` | Continue bounded search |
| [Five words](receipts/6f806410852b4ff0a3482569b3b3fc99.json), 20,001 updates | 0/5 canonical words; quality `0 < 0.95`; TV `1 > 0.1`; exact inversion `0`, required 1; minimum correct inverse probability `0 < 0.9` | Completed budget; 1,024 samples | `4/24`; `0` | Continue bounded search |

All declared optimizer roles completed their budgets, states remained finite,
and the recorded RNG audits report zero unintended deviations. These are
scientific failures, rather than incomplete training or infrastructure errors.
[Compact numerical and provenance extract](TIER1_FAILURE_REVIEW_EVIDENCE.json).

## Gaussian: acquisition happens, then location and width oscillate

The target is `N(2, 0.5²)`. BCAP passes every bound at updates 667 and 959, but
does not retain the solution. Its last five checks show the instability:

| Update | Mean error / target σ, ≤0.2 | Width ratio, 0.8–1.2 | CDF KS, ≤0.05 |
| ---: | ---: | ---: | ---: |
| 834 | 0.275592 | 1.210137 | 0.117652 |
| 875 | 0.125346 | 0.937554 | 0.065166 |
| 917 | 0.176031 | 0.706228 | 0.143440 |
| 959 | 0.035039 | 1.017563 | 0.046392 |
| 1,000 | 0.007826 | 1.261683 | 0.096744 |

The final mean is excellent, but the final width is too large; just two checks
earlier it was too narrow. Calling this an inability to move toward the target
would miss the measured problem: **the acquired distribution does not settle**.

The [original lower-rate relativistic run](receipts/ad6b7c03424542d3b6090541b9923ae9.json)
at LR `0.0010625` improves stability of the terminal moments. It passes 6/24
checks, but finishes at KS `0.068207`, with suffix 0. Its last five KS values are
`0.054118, 0.082311, 0.049318, 0.038506, 0.068207`. Lowering the whole rate helps
some aspects; it is not a complete solution.

**Likely direction, not established cause:** constant G/D rates of `0.00425` and
prior rate `0.0085` may sustain adversarial equilibrium motion. Role-rate balance
is an untested factor in the completed pure-BCAP search. There is no saved
critic-gradient or cap-activity history sufficient to identify which role or
penalty behavior caused these oscillations. More unchanged updates might extend
the oscillation; they are not an evidence-backed repair.

**Why not loosen?** The [same-moment uniform control](../../../reports/toy_audit/api_contract/gaussian1d/controls.json)
has the correct mean and variance, but KS `0.057385`; only the shape gate rejects
it. Raising KS enough to accept even the lower-rate endpoint would admit that
non-Gaussian law. Accepting the incumbent's whole terminal window would require
KS at least `0.143440`, width limits at least as broad as `0.706228–1.261683`, and
mean-error tolerance at least `0.275592σ`. That changes the scientific question
substantially. Disabling the cheap scalar test would remove the direct check
that acquisition preserves Gaussian CDF shape rather than just two moments.

## Ring: finding the modes is insufficient; tails still fail badly

The target has 16 equally weighted Gaussian clusters, radius 3 and component
σ `0.1`. The endpoint reaches every mode and acceptable mass balance. Its full
component covariance error is nevertheless **6.3 times the allowed maximum**.
The preceding checks are worse:

| Update | Modes, required 16 | HQ, ≥0.85 | Full component covariance error, ≤0.85 |
| ---: | ---: | ---: | ---: |
| 334 | 10 | 0.469238 | 11.114540 |
| 350 | 15 | 0.605225 | 10.879496 |
| 367 | 10 | 0.495605 | 9.122850 |
| 384 | 16 | 0.731934 | 6.325117 |
| 400 | 16 | 0.822754 | 5.366516 |

The endpoint has **726/4,096 samples outside the nearest component's 3σ region**.
Its core covariance diagnostic is `0.501662`, much better than full covariance
`5.366516`, and maximum component spill is `0.345178`. Good central regions
coexist with poorly assigned tails/bridges. Substituting core covariance or
global ring covariance for the full-component gate would hide that error.

BCAP's [soft cap](../../../particlegan/grad_regularizers.py) penalizes critic
input-gradient norms only above the cap. It does not directly constrain
generator covariance, prior transport or tail mass. Those are plausible places
to investigate, but 400 recorded penalty applications do not prove the cap was
active on the important samples. The original vector task saves observations,
not a critic/optimizer checkpoint, so a causal critic-state audit cannot be
invented from this archive.

**Why not loosen or disable?** Lowering only HQ to admit `0.822754` does not fix
the covariance failure or the four earlier checks. Admitting the entire terminal
window would also accept 10 modes, HQ around `0.469`, and covariance error above
11. The [independent target control](../../../reports/toy_audit/api_contract/ring16/controls.json)
passes with HQ `0.989746` and covariance error `0.081982`; destructive controls
reject missing modes, collapsed spread and other errors. The distinct
[selected K3P ring run](../tier1-completion/README.md) passes the same host's full
terminal gate. That makes a universal architecture/budget impossibility claim
untenable, without isolating the BCAP penalty as the cause.

Retain the 400-update allowance for the first targeted successor. Extra settling
for **pure** BCAP remains an untested, separately scoped budget hypothesis.

## Words: catastrophic loss of a previously correct solution

This task learns a uniform five-word vocabulary together with its inverse map.
Its explicitly declared prior is **five learned 2D particle-cloud rows**, σ 0;
it is not the Gaussian/ring MoG cohort. All generator, encoder, prior and critic
roles complete 20,001 updates.

The first four checks, at updates 834, 1,667, 2,501 and 3,334, pass every bound.
By 4,167, canonical quality and mode count are zero, and remain zero through the
endpoint. The loss of the solution therefore occurs between 3,334 and 4,167:

| Update | Canonical modes | Quality | Mass TV | Exact inverse | Minimum correct inverse probability |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 834 | 5 | 1 | 0.018945 | 1 | 0.995664 |
| 3,334 | 5 | 1 | 0.035547 | 1 | 0.997494 |
| 4,167 | 0 | 0 | 1 | 0 | 0 |
| 20,001 | 0 | 0 | 1 | 0 | 0 |

The hash-verified saved outputs are confidently malformed, rather than slightly
below a confidence threshold. At the endpoint every generated token's maximum
probability is 1; decoded outputs are `beeoia` (621), `bekoid` (203), and `bekfed`
(200). The model has saturated on incorrect strings. Lowering the `0.90` inverse
confidence threshold would not restore canonical generation or inversion.

There is a useful original-source control: the
[lower-rate pure relativistic recipe](receipts/a36a9ee30bfc4caa92fa9f842613c2e4.json)
passes with suffix **8**, five modes, quality 1, TV `0.018945` and minimum correct
inverse probability `0.998710`. The
[corrected low-rate non-saturating recipe](receipts/a8a570a740f4420a81403d8e077b6b16.json)
also passes, with suffix **22**. The lower relativistic recipe has only 2/6
required passes overall; its word cell cannot be grafted onto the incumbent.

The original nonrelativistic missing-encoder-loss bug is distinct from this
failure. The selected relativistic formula already trains both joint streams;
the [exact-trajectory software control](../../../tests/test_pure_bcap_joint.py)
checks that the corrected joint-loss API preserves its previous behavior.
The [word root-cause study](../word-root-cause/README.md) supplies further
attainability and implementation context, but its scheduled K3P/KA2 noise and
annealing explanations do not apply to constant-rate, noise-free BCAP.

**Likely direction:** high constant rates destabilize a learned joint
equilibrium. Collapse and saturation are observed; critic dominance, prior
drift or overshoot are unisolated explanations. Twenty later failed checks
give little support for simply extending this same training law.

**Why not loosen or disable?** This is a maximal endpoint failure on five
substantive bounds, and multiple BCAP recipes meet the original word contract.
Selecting an early checkpoint would define a different serving/selection law
and a transient-acquisition question. Disabling the test would remove joint
categorical and inverse-map coverage.

## What we should do next, and when to stop

The [completed comparison](readout.json) already covers **five losses × two
global rates**. All ten recipes fail Gaussian and ring; none passes the six
required tasks. It charged **2,549.509 seconds across 109 unique paid attempts**,
including the original cancelled/bugged work and corrected round. Those costs
overlap the other publications and must not be added again. Repeating that grid
or changing only the seed is unsupported.

1. **Finish a zero-training anatomy review before choosing values.** Use the
   saved word endpoint and existing observations to inspect saturation and
   loss of correctness. That checkpoint cannot identify which role moved during
   the 3,334–4,167 collapse interval; its parameter trajectory is unavailable.
   Gaussian/ring critic states are also absent. During a new registered comparison, collect cap-active
   fractions, critic input-gradient norms and actual role updates through
   diagnostics that preserve model state and consumed RNG streams.
2. **Freeze one finite role-rate balance study.** Keep native Adam, constant
   rates, relativistic loss and the cap mechanism. Start from the incumbent base
   LR for successor recipes and change only declared critic/prior role
   multipliers, applied globally across tasks. Include the lower global-rate
   recipe as an explicit matched control if compatible evidence is unavailable.
   The historical low-rate comparison slows G, D and prior together: a fixed-G-LR
   ratio study tests role imbalance, and cannot rule out excessive common/global
   step size. Ring and short direct-coordinate movement constrain slowing.
   Do not choose separate recipes for individual tests. If the anatomy instead
   supports cap pressure, declare that factor group as a separate study rather
   than stacking changes or launching a broad sweep.
3. **Cap the comparison at four whole recipes, including any needed matched
   baseline.** Six required tasks reserve 2,220 seconds per recipe; retaining
   the 300-second clock diagnostic gives **10,080 seconds maximum reservation**
   for four recipes: a matched incumbent, a low-global-rate control and at most
   two role-ratio successors when neither control can be reused. Register a new
   study/campaign; concluded campaigns do not replenish themselves. Complete
   independent peers in the current tier after a failure. This study ends at
   Tier 1 regardless of outcome; later-tier work requires all lower-tier passes
   and a separately declared study and budget.
4. **Judge one whole recipe.** A successor must pass all six required tests and
   retain the pure-family clock result. Publish every outcome. If no recipe
   clears the tier, close the study and stop routine parameter search; require
   a new measured mechanism hypothesis before spending again. Keep BCAP as an
   explicitly limited baseline rather than manufacturing a universal winner.

All new comparisons must use protocol seed 0, the **current public deterministic
initializer**, identical per-task architectures/data laws/initial model states,
seen batch sequences, priors, sampling, budgets and evaluation cadence, with
isolated constructor/data/training-noise/evaluation RNGs and checkpointed streams.
Reuse baseline evidence only when every scientific binding matches. Modern
initialization or batch-stream changes define a new cohort; they cannot silently
replace or regrade these archived results.

## When the other two options would be justified

**Loosen requirements only after independent calibration changes our scientific
claim.** The [experiment rules](../../../EXPERIMENTATION.md) explicitly keep the
screen provisional. Oracle/destructive scorer controls are not calibration of
trained false rejection or Tier 1 placement. If independently useful trained
references systematically fail these cheap gates while passing downstream
quality, review the task's placement, update budget or evaluation precision in a
new frozen protocol. Gaussian finite-sample uncertainty near the KS boundary
merits such a precision review; it does not explain away the incumbent's width
oscillation or justify admitting the uniform control. Preserve all original
verdicts. A future Tier 1 reclassification would not be a trained-quality PASS.

**Disable a test only when its question is redundant, invalid or outside the
claimed trainer scope.** Repair a defect or register a narrower runnable variant
first, retaining useful controls and original evidence. No reviewed evidence
establishes those conditions for these three tests. For an explicitly
GAN-only product that makes no joint/inverse claim, a view omitting words could
be honest scope; it would not qualify BCAP for the current full claim. Merely
removing the failing cells would make the board greener without answering them.

Even a future passing screen would need calibration and separately registered
confirmation before public-default adoption.

## Evidence identity and verification

The incumbent is archived under commit
`af75a3fea19aa6e4d1ca2be867b9c50a02931e33`, source digest
`eede2a3a5780805489664b7354e8748a4020fa8eb5023aa6cad50c8e3f9d05ac`,
candidate revision
`9139a994e2a88403955e5bc8a0ce95be8c2cd5be8f0b2fad0652350d24b6971c`.
Its seed-0 named orthogonal/constructor initialization is preserved. The eight
corrected nonrelativistic recipes bind commit
`44cc66d78495cb913e0ea064840e2de37c8a4ae5`; the ten-recipe readout is not one
merged source cohort. Today's word definition is marked **changed since run**;
that does not erase its original failure or establish a current-source result.

The [review extract](TIER1_FAILURE_REVIEW_EVIDENCE.json) binds final numbers,
convergence, sparse terminal evidence, raw-result hashes and the word tensor
identity. Original local raw results were compared byte-for-byte with members
of the hash-verified [original archive](archive-initial.json). The
[complete-round archive](archive.json) retains the corrected-source evidence.
These extracts are display-only; original envelopes are required for regrading.
No bulk execution log, checkpoint or observation stream is added to Git.

Existing actual-training illustrations remain available for
[Gaussian](media/5ea5bdbb2d71/gaussian1d_acquisition.gif),
[ring](media/5ea5bdbb2d71/ring16_acquisition.gif), and
[words](media/5ea5bdbb2d71/five_word_joint_acquisition.gif).
The numerical evidence drives this recommendation; the GIFs illustrate it.
