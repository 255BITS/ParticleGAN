# Bregman game: measured readout

`bispf-bregman-game-v1` finishes **6 PASS / 9 FAIL** across all fifteen unchanged tasks. It repairs four original failures: overlap, bars, blobs and intensity. It also loses two original controls, ring acquisition and mode hold, retaining **2/4 controls**. The fresh energy reference also passes six tasks and retains all four controls; the game additionally loses its anisotropic pass. **Reject a global replacement or qualification.** Preserve the measured image/overlap improvements as research evidence. No taskwise mixture of recipes is selected.

The [campaign comparison](../README.md) owns the single current leaderboard, scientific admission, runtime/initialization/stream audit, paid-cost ledger and actual-training GIF publication. This report explains the complete game arm against both fresh fifteen-task references. The frozen [mechanism, predictions and cheap controls](README.md) remain unchanged.

## Contract and complete task results

The sole global recipe delta from `bispf-optimized-quintic-v1` is `bispf_bregman_game: false → true`; auxiliary energy/quartic transport remain zero. This replaces both adversarial payoffs with the learned hard Bregman c-transform game and uses the declared identity/sigmoid output link. It preserves the task models, data laws, protocol seed 0, public initializer, initial prior, capacity, consumed sampling streams, regularizer, prior update, optimizer, update budget and evaluation schedule. The original BISPF fake penalty and thermal hook still run. This is a finite-batch game with one neural price update per step, not a converged population transport solve.

| Task | Game gate | Complete curve / final requirement |
|---|---|---|
| gaussian1d_smoke | PASS | Original acquisition criterion: 5/24 passing checks, first confirmed at 167. |
| gaussian1d_stability | FAIL | Stationary 10/72, frozen 0/48, shifted hold 2/24; reacquisition fails. |
| vector_unequal_mass | FAIL | 0/24 joint passes; rare 2% component absent. |
| vector_unequal_width | FAIL | 12/24 joint passes; longest run 5 is transient; terminal suffix 0. |
| vector_anisotropic | FAIL | 4/24 joint passes, terminal suffix 0; component shape fails. |
| vector_overlap | PASS | 21/24 joint passes, terminal suffix 9; confirmed at 1000. |
| img_bars4 | PASS | 6/24 joint passes, terminal suffix 6; confirmed at 575. |
| img_blobs4 | PASS | 8/24 joint passes, terminal suffix 8; confirmed at 525. |
| img_intensity2 | PASS | 10/24 joint passes, terminal suffix 9; confirmed at 500. |
| grid100 | FAIL | 0/34 coverage checks, 0/5 terminal accuracy checks; 100k holdout fails. |
| rotated100 | FAIL | 0/34 coverage checks, 0/5 terminal accuracy checks; 100k holdout fails. |
| staggered100 | FAIL | 0/34 coverage checks, 0/5 terminal accuracy checks; 100k holdout fails. |
| ring16_acquisition | FAIL | 0/96 joint passes; full covariance and HQ fail. |
| mode_hold | FAIL | 1/24 joint passes at step 400, terminal suffix 0; only 3/8 modes remain. |
| img_stripes2 | PASS | 11/24 joint passes, terminal suffix 11; confirmed at 450. |

Acquisition and continuation retain their original phase rules. In particular, smoke's endpoint KS is .20793 despite its earlier acquisition PASS; it is not a new endpoint-only gate. The continuation failure prevents treating acquisition as durable Gaussian fidelity. The full compact evaluator summaries retain every original bound, phase count and terminal status.

## Images: clean modes and local pixel fidelity improve together

The game repairs the original clean-mode gates, rather than merely balancing nearest-template assignments. The final bars clean counts are `[8,8,7,7]`, blobs `[7,8,8,8]`, and intensity `[16,16]`. Their original HQ values are .9375, .96875 and 1.0, with all 4/4/2 modes present. They pass the required terminal five-check criterion. Original bars had only two clean modes; original intensity already had balanced nearest assignments but only 19/32 clean images.

| Final saved image statistic | Original | Energy | Game |
|---|---:|---:|---:|
| Bars mean pixel MSE | .00382489 | .00790796 | .00131623 |
| Blobs mean pixel MSE | .01171015 | .00195302 | .00122340 |
| Intensity mean pixel MSE | .00453005 | .00552239 | .00011673 |
| Intensity patch-mean MSE contribution | .00161215 | .00194450 | .00005805 |
| Intensity within-patch MSE contribution | .00291722 | .00357732 | .00004498 |
| Intensity background MSE contribution | .000000683 | .000000574 | .000013693 |

These supplementary MSE values average squared error to the nearest fixed template over every saved pixel; they do not replace the original per-image RMSE/HQ gates. Intensity improves patch mean and spatial variation substantially while its small absolute background error increases. Its within-patch error fraction is .3854, versus .6440 originally; the absolute MSE decomposition above avoids interpreting a changing denominator as a gain. Shared spatial-pattern energy fractions fall from .731/.877 to .138/.126 across the two levels.

All 600 game training updates on each of bars/blobs/intensity have finite diagnostics, zero exact output endpoints, zero near-saturation fractions at ≤1e-6 or ≥1−1e-6, and no prelink magnitude ≥16. Across those updates the extreme prelinks are bars [−10.390,12.859], blobs [−10.438,7.605], intensity [−7.752,3.467]. Final saved evaluation pixels likewise have no exact endpoints. This is consistent with the stable sigmoid pullback avoiding an extreme saturation trajectory from the shared initialization. It does **not** demonstrate recovery from a checkpoint that was already saturated, nor identify which of changed pricing, pairing and link pullback caused the gains: they changed as one registered mechanism.

The healthy stripes control still passes, but its original mean RMSE worsens from 3.74e−5 to .006156. Stripes does reach exact one-valued pixels; absence of exact endpoints is not itself a success gate.

## Width tails improve, but local covariance and retention remain failures

Width's original full component covariance error 9.68690 becomes **.488562**, with core error also .488562. Its mass TV improves from .277588 to .006592, HQ reaches .999268, and normalized sliced Wasserstein becomes .045033. The final saved samples have **zero >4σ tail points in all four components**. This is real full-law tail improvement, not a core-only substitution.

The remaining endpoint minimum eigenvalue ratio is **.147383 < .15**. Assigned component trace/target ratios are [.23389,.32396,.80145,.71472], showing substantial underdispersion, especially in the narrow components. Their mean displacements are [.09815,.04596,.14835,.02008]. The curve passes jointly 12 times and once has five consecutive passes, but ends with suffix zero; the original requirement is a final stable suffix. The tail problem is reduced without solving persistent local shape. These are served saved samples only; no new center-output forward was added, so this analysis cannot split center and kernel contributions.

Unequal mass still has final assignments `[1896,1614,586,0]`: the 2% component is entirely missing. Full/core covariance errors are 1.87674/1.13438, HQ .73926, minimum eigen ratio zero. One populated component has 889 >4σ points and 36.5% of its centered scatter outside the core. Anisotropic has full/core errors .91194/.91348 and minimum eigen ratio .004879; its component trace/target ratios [.12853,.20460,.06896] show severe underdispersion. Energy instead passes anisotropic with full error .16720 and minimum eigen ratio .47311. Better global occupancy is therefore insufficient evidence for local geometry.

Overlap is a sustained success: final covariance error .259825 ≤ .45, mean error .059326 ≤ .15, normalized sliced Wasserstein .080468 ≤ .18, with terminal suffix 9. Energy has a better endpoint sliced distance .052610 but suffix 1, hence still FAIL under the unchanged gate.

## Native precision deteriorates despite improved nearest-cell mass

| 100k clean holdout | Original precision | Energy precision | Game precision | Game mass TV |
|---|---:|---:|---:|---:|
| grid100 | .30497 | .32545 | **.07912** | .03085 |
| rotated100 | .22301 | .24826 | **.06664** | .06279 |
| staggered100 | .33617 | .29757 | **.11210** | .04194 |

These are large continuous regressions on tasks that already fail in the references; a PASS→FAIL list alone would hide them. All generated values remain finite. Nearest-cell mass is substantially closer to uniform, yet high-quality density near the narrow target components deteriorates. That observation does not establish point collapse.

The frozen coverage gate still requires precision ≥.97, all 100 modes, high-quality mode mass ≥.005, mass TV ≤.10, maximum mode mass ≤.02, covariance eigen ratios [.4,1.7], radial-median ratio [.65,1.4], and finite samples. Each final five-check cloud contains 20k samples. The independent 100k holdout and original accuracy bounds remain mass TV ≤.06, center RMS ≤.20 target σ, absolute covariance trace bias ≤.10, and radial KS ≤.04. No curve check passes coverage or terminal accuracy. Grid/staggered satisfy the *holdout mass-only* bound; rotated's .06279 does not.

All five terminal checks and holdouts have undefined conditional center/shape statistics for all three game natives. The original estimator returns null when any component has fewer than two in-radius (3σ) samples. These nulls cannot certify either centering or covariance; they are retained as failure evidence. `center_max_sigma` and covariance Frobenius RMS are diagnostics, not independent accuracy gates. The compact receipt binds the original task declarations and accuracy source hashes.

## Prices, routes and canonical force do not certify distribution matching

All **37,000** recorded game updates have finite returned diagnostics. Thermal scale is exactly 1 throughout, so the analytic forces before and after thermal coincide. There is no observed thermal attenuation to explain these failures.

| Last-decile mean | Bars | Blobs | Intensity | Grid100 |
|---|---:|---:|---:|---:|
| Selected Bregman cost | .102661 | .021226 | .003426 | .001770 |
| Real-row price standard deviation | .885594 | .102614 | .021296 | .008204 |
| Used real-row fraction | .17240 | .22969 | .17135 | .40324 |
| Route entropy, nats | 1.54303 | 1.75686 | 1.38362 | 6.41137 |
| Canonical force RMS | 4.9773e−5 | 3.1100e−5 | 8.1169e−6 | 5.0546e−6 |
| Canonical force maximum | 2.8517e−4 | 3.5499e−4 | 6.9949e−5 | 1.7984e−5 |

The force is the analytic negative derivative of the **mean adversarial generator payoff** with respect to identity output or the actual sigmoid prelink, including the batch/coordinate means and identity real-variance normalization. It excludes the unchanged prior regularizer and is **not realized parameter motion** or an inverse-Jacobian measurement. The game diagnostics describe the G-phase batch before its move and are returned after the public update. Scored saved images use separate post-update evaluation samples; this is not a pointwise force/error pairing. Cross-task force magnitudes also differ with batch size, dimension and cost units.

The row fractions are not mode coverage: all four blobs modes can be healthy while only about 23% of the repeated real batch rows are selected. Grid's selected cost falls from first-decile .04145 to last-decile .00177 while its precision remains poor; global variance normalization and learned nearest-row routing do not certify narrow component accuracy. Its price spread falls from .12074 to .00820, but a small spread does not prove dual convergence. Ring similarly has low final-decile cost .001682 and canonical RMS .000105 while failing full covariance. For unequal mass and anisotropic, used-row fractions fall from .303/.297 to .0389/.0379 even as price spreads rise to .383/.256; the witness is active without delivering the required rare/shape fidelity. These observations are consistent with finite-batch routing/price dynamics failing to enforce every local law. They do not isolate a unique causal bottleneck.

## Controls reject global promotion

Ring preserves all 16 modes and improves mass TV to .024658, but HQ **.844971 < .85** and full component covariance error **1.385529 > .85** fail. Its core-only error .898424 also does not rescue the full gate. The original and energy terminal suffixes are 12 and 47, versus zero game passes across all 96 checks. This is a fidelity/retention regression despite improved mass.

Mode hold briefly passes once at step 400 and ends with **3/8 modes, HQ .315430**, against required 8 modes and HQ ≥.9. Original/energy suffixes are 10/7; the game suffix is zero. Gaussian smoke and stripes are the two retained controls. Thus matching the energy arm's six total passes does not make the game comparably suitable as a global trainer. The image gains remain useful evidence for the registered paired payoff and stable output-link formulation; they do not justify relaxing failed vector/native/control gates. No additional paid successor was launched from this readout.

## Evidence and reproduction

The [compact saved-only receipt](measured-analysis.json) contains final metrics, aggregate whole-curve ranges/counts, phase/terminal summaries, hashes/paths to immutable attempts, certificates, saved tensors and original public traces. It contains no copied per-observation curves or per-update event streams. Those bulk artifacts remain under `/mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022`.

Scientific execution source: `a03373da3421303b89c99a39898b091e22910134`; source digest: `668914920afda7bb6fc53d9c918aac92a9c9cd5ae582884fb9f068ff1b8249e8`. Complete full-readout SHA-256 identities are:

- Original: `dd7528d742fb790ae1eb8330d2c00cf9d9d184935653d643a042ee04f08c9794`.
- Energy: `e5d24339debeec1fe9a36e1290cac7cd1ab82c9f0bb5999567b2a4bb9f6afac8`.
- Game: `b5d08a18fd002d70f7e30b937d8b0250e7e7c4a8551e6dde72cda0e303741a49`.

Run [reduce_measured.py](reduce_measured.py) only after all three full readouts are complete:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python reports/forge/first-principles-round2-20261022/fidelity/reduce_measured.py
```

The reducer verifies all fifteen unique task cells per arm, source identities, saved-output hashes and contiguous complete trace steps before reducing. It constructs no models and adds **zero model forwards, sampling draws or optimizer updates**. The completed reduction used 2.93 CPU seconds (2.96 seconds wall); its stdout/timing log is archived at `fidelity-analysis.log` in the raw campaign root. The analysis JSON SHA-256 is `28b8372309a99d3b30a2809e5cb1b28df4d41dc1e686191297f7a2741c76c20b`. This is a saved-state research analysis, not ordinary qualification or default promotion.
