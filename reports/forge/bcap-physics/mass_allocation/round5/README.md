# Round five: conservative joint transport repairs width, loses rare density

**Balanced joint-coordinate transport repairs unequal width under the full sustained gates, but loses local-v2's rare-density repair and worsens native genuine quality mass.** Both whole configurations finish **3 PASS / 3 FAIL** across the six declared tasks. Retain the width result as a scoped experimental reference; retain local-v2 for rare density. Stop this exact candidate as a global replacement. The provisional diagnostic supplies no ordinary qualification or default promotion, and the [single current technique inventory](../../../technique-inventory.md) retains its original winner.

## Completed matched comparison

| Unchanged task | Exact local-v2 control | Balanced assignment candidate | Measured consequence |
| --- | --- | --- | --- |
| Gaussian smoke, 1,000 updates | PASS, 13/24 confirmed states | PASS, 13/24 confirmed states | Both first confirm at 167; endpoint KS .020606 |
| Own Gaussian stability, 5,000 additional updates | FAIL | FAIL | Stationary 28/72→33/72; shifted 11/24→12/24; deadline reacquisition fails both; final KS .060653→.031027 |
| Unequal mass, 1,200 updates | **PASS**, suffix 5 | **FAIL**, suffix 0 | Full covariance .522577→1.099564 exceeds .85, despite mass TV .016426→.011299 |
| Two broad, 1,200 updates | **PASS**, suffix 24 | **PASS**, suffix 24 | Full covariance .226564→.213764; broad guardrail retained |
| Unequal width, 1,200 updates | FAIL, 0/24 checks | **PASS**, 14/24 checks and suffix 14 | Full covariance 2.480438→.286062, HQ .955078→.978027; all four components remain populated |
| Native grid100, full 7,000 updates | FAIL | FAIL | Holdout precision .219730→.152760, genuine quality modes 17→9; all five terminal checks fail |

[Final metrics and temporal gate failures](results.json), [compact original receipts](receipts.json), [matched protocol proofs](provenance.json), [frozen-source/data audit](validation.json), and [actual endpoint center populations](center-populations.json) bind this comparison. Native holdouts use 100,000 clean/live public-prior draws; their metrics are separate from the 20,000-draw scheduled observations. No earlier state replaces a failed terminal window.

**Width improves center-population shape.** The narrow components' full served covariance errors fall 4.787480/4.601254→.374944/.124869. Candidate mass TV is .015625 and the minimum component eigen ratio .476344 passes the .15 floor. Saved uniform center counts are 61/64/65/66→62/65/65/64, while narrow center-population covariance traces fall 5.1229/4.3290→.7219/.8891 times target trace. These are exact `G(z_location)` endpoint populations with diagnostic nearest-center assignments. They exclude nonlinear within-kernel jitter and do not establish full-history attribution; the served-law sustained PASS is authoritative.

**Rare allocation improves while rare spread worsens.** Center-population TV falls .005000→.001563, with counts 141/77/32/6→141/77/33/5. The rare center-population trace rises .7757→3.7285 times target trace. The candidate passes only 5/24 observations; terminal covariance errors are `.903159,.742228,.958393,.975773,1.099564`, and update 1100 also fails minimum eigen ratio with .041922. Better allocation or endpoint mass cannot substitute for the original full shape and five-check retention requirements.

**Native covariance reduction hides worse genuine quality.** Both holdouts occupy all 100 nearest cells. Candidate center-population TV improves .08455→.07810, and served mass TV .08646→.08002 remains above the accuracy bound .06. Uncensored covariance Frobenius RMS falls 35.3650→9.0178, yet spill rises .78027→.84724, minimum genuine quality mass falls .00001→0, center RMS rises 5.5624→6.7013 target sigmas, and mean full radial KS worsens .8280→.8726. Only 9/100 modes reach genuine mass .005, versus 17 for control; all 100 are required. Nullable official native shape metrics remain null where their quality cohort is insufficient. Supplementary uncensored moments do not rewrite those gates.

**Gaussian endpoint improvement is not retention.** Final candidate KS .031027 passes the individual .05 bound, but the complete stationary/shift/reacquisition gate fails. Smoke observation curves are bitwise identical across these two arms. Their stability curves first differ at 1959 despite matching consumed target sequences. In one dimension both mathematical costs are empirical quadratic transport; different finite numerical implementations/tie branches need not preserve a long optimization path. This comparison does not isolate a scalar improvement caused by joint-coordinate geometry.

## Mechanism and saved motivation

The [preregistered endpoint census](prior-diagnostics.json) separates center occupancy from served quality. The saved native winner has center TV .1459 and served TV .1480 in its 20,000-draw endpoint cohort, yet only 9/100 diagnostic three-sigma modes reach half their target mass. Local-v2 already preserves rare and broad PASSes, so it is the sole matched primary control; the original winner remains source-bound context.

For 128 saved served rows paired with the replayed final target-batch prefix, diagnostic cross-component pairings decrease under joint assignment versus sliced pairings: native .9224→.7813, rare .0984→.0469, broad .0991→.0313, width .2087→.1094. The conservative flux table is an output-space coupling, not original latent rows, measured G/prior pullbacks, or historical training flux. Labels, target means and covariance enter these diagnostics only.

The receipt also retains the exact archived finite-role probe source and checkpoint identities from [round-four role motion](https://github.com/255BITS/ParticleGAN/blob/871b76af18388e70595db1eb9d685f7a8de8af70/reports/forge/bcap-physics/role_motion/round4/README.md). That saved native next proposal has network/prior/joint output RMS .245671/.036805/.268698, with network center travel .245057 versus jitter deformation .005016. This one restored endpoint probe is contextual, not a new update in this campaign or a training-history attribution. The [latest root round-four report](https://github.com/255BITS/ParticleGAN/blob/c2fb4ab1d249a4e053220e8acf9f625e35af3cf5/reports/forge/bcap-physics/round4/README.md) and [read context hashes](research-context.json) preserve the prior conclusions.

The **one trainer delta** is `kinetic_transport_mode: sliced→balanced_assignment`. Public `Recipe.kinetic_transport_loss` selects reusable `balanced_assignment_loss`; weight 1 and exact local-v2 local moments weight 1 remain unchanged. For each contiguous block `B`, solve the squared-coordinate bijection `pi_B` minimizing `sum_i ||x_i-y_pi(i)||²`. Every fake and real row participates once; the final partial block retains all its rows. The total loss is

`L_assignment = sum_i ||x_i-y_pi(i)||² / (N * dimension * detached real coordinate variance)`.

Assignments use detached float64 CPU costs and SciPy; gradients flow through the selected Torch costs. Block size 128 is one global configuration: one block for each vector/Gaussian batch and 16 blocks for native 2048. This is exact empirical squared W₂ within each block, not full-batch OT across blocks. There is no sampling, prior-weight, width, routing, acceptance, oracle-label or persistent-state change. The same real/fake tensors already consumed in the G phase supply the loss; frozen vector/Gaussian hosts retain their actual same-real-tensor D/G law.

[Fatras et al.](https://proceedings.mlr.press/v108/fatras20a.html) analyze minibatch OT and its loss of the population distance property. A conservative finite coupling does not guarantee population fidelity; minibatch mass fluctuations and shared-map response remain competing explanations for the rare regression. [SciPy's solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) supplies the square minimizing assignment; ties select one branch. SciPy is the existing optional `mog` dependency, and missing support blocks preflight before reservation. No formal or analogy-based convergence claim is made.

Both arms retain nonsaturating loss, full DualNorm momentum 0/smoothing .001/per-offset layout, G/E .012, D .018, prior .030, constant floors 1, BCAP coefficient 1/cap 1/every update, zero extra prior regularization, additive noise and EMA. Public deterministic initialization, seed 0, each task's architecture/data law, consumed batches, learned MoG locations/uniform masses, sampling, committed-update budgets and scoring cadence are fixed. Prior widths retain their task declarations: .025 for native/vector and .1 for Gaussian. No task card or threshold changes. Gaussian continuations restore their own complete 1,000-update state and every consumed stream; failed own smoke remains authoritative. Fixed conditional/two-pole fixtures and other hosts are outside this declared subset.

## Execution, repair and validation

[Two ready schema-v3 arms and finite studies](freeze.json) were pushed before training at scientific commit **`6e0c08dfe22e4bb57624d3211e67ef7cc07de8a5`**, source digest **`3b09403805f51c1dc8dc7a520eef7de0879292f7d5e595640842fc968c10dbd3`**. All matched arms exercise that frozen source/runtime. Later edits are reporting/reproduction only; measured trainer bytes remain unchanged.

All 12 task jobs finish with no final BLOCKED/INVALID/INCOMPLETE result. There are 13 paid attempts including one preserved **INCOMPLETE** smoke timeout: training recorded 1,000 updates, but the supervised attempt exceeded its 120-second allowance. The [execution-only repair](execution-repair.json) retains its original receipts and **121.387861** paid seconds, and links the same frozen request's successful retry after returning to one active worker. The two-device coordinator experiment respected host-memory admission; it created no alternate scientific arm. Original timeouts are not rewritten or counted as PASS. Both coordinators and the log follower have stopped.

Total paid wall time is **2,963.087384 seconds**, 49.38 worker minutes, including that timeout. Original full allowances total 19,440; executed full allowances including the 120-second retry total **19,560**, plus separate 120-second saved-diagnostic, synthetic-capacity and software-fixture allowances, within the **21,600** track ceiling. Remaining reservation is 0. There are 33,200 certified committed training updates, plus 1,000 recorded in the retained timeout attempt. Contention/setup/CPU solving are included in their actual scopes and establish no speed ranking. [Capacity check](capacity.json) uses synthetic 2048-row forward/backward losses only, with zero trainer updates.

**36 meaningful mechanism/protocol checks pass**, with 10 public-trainer updates in an explicit separate synthetic linear fixture cohort across initial/final checks. The audit verifies 2,406 frozen source files across both requests, unchanged measured implementation/declarations and original task/qualification/telemetry hashes, matching initial model/prior hashes and consumed named training streams. Four complete native/vector target sequences replay to both actual final RNG states. Both Gaussian continuations certify exact own-parent state and no history reset. [Archived singleton parity](control-parity.json) verifies bitwise local-v2 metrics, observations and numerical state for smoke/stability/rare/broad; it supplies no new qualification reuse. Publication recomputes 432 saved scalar/vector metric sets and adds zero optimizer updates or public samples. [Scorer controls](scorer-controls.json) pass all five oracle controls and reject all ten destructive controls, including balanced point collapse.

The [publication checks](publication-verification.json) confirm Forge validation, summary freshness, all 88 preserved task/qualification/publication files, all 414 measured working implementation/declaration files, 12 nine-frame GIFs, and stopped task workers. The [administrative stop](../../../records/lifecycle-3adc8030026cb516f274d123.json) binds the candidate's exact revision and complete readout; its scoped width evidence remains available.

Bulk stdout, JSONL, checkpoints and tensors remain on the artifact drive. The one current goal leaderboard is unchanged. The report preserves the width success, rare regression, native late failure and Gaussian limits together; no sweep, second candidate, seed study, extended budget, merge or promotion follows.

```sh
# Compact live task summaries and exact driver events:
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/mass_allocation/logs/progress.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/mass_allocation/logs/driver.log
# Read-only reproduction from retained artifacts:
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/mass_allocation/round5/publish.py
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/mass_allocation/round5/audit.py
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/mass_allocation/round5/analyze.py
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/mass_allocation/round5/parity.py
```

## Actual-training media

| Original full task | Local-v2 control | Balanced assignment |
| --- | --- | --- |
| Gaussian smoke | [GIF](media/control-gaussian1d_smoke.gif) | [GIF](media/candidate-gaussian1d_smoke.gif) |
| Gaussian own stability | [GIF](media/control-gaussian1d_stability.gif) | [GIF](media/candidate-gaussian1d_stability.gif) |
| Unequal mass | [GIF](media/control-vector_unequal_mass.gif) | [GIF](media/candidate-vector_unequal_mass.gif) |
| Two broad | [GIF](media/control-vector_two_broad.gif) | [GIF](media/candidate-vector_two_broad.gif) |
| Unequal width | [GIF](media/control-vector_unequal_width.gif) | [GIF](media/candidate-vector_unequal_width.gif) |
| Native grid100 | [GIF](media/control-grid100.gif) | [GIF](media/candidate-grid100.gif) |

[Media certificates](media/index.json) bind all 12 GIFs to saved scored observations, fixed nine-frame index selection, original numerical cadence and source inputs. They illustrate the actual training goal; numerical gates decide the outcomes.
