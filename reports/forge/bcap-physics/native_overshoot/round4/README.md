# Native finite movement, round four

**Armijo catches actual overshoot and improves native precision, but does not repair the full density law. Stop this exact revision.** Holdout precision increases **.24072 → .65938**, while genuine quality mass, mass balance, local covariance, radial shape and the complete terminal window still fail. Gaussian stability also fails. The matched control remains the exact archived BCAP winner; there is no ordinary Tier2 qualification or default promotion.

| Unchanged full task | Exact winner control | Armijo candidate | Evidence |
| --- | --- | --- | --- |
| Gaussian smoke, 1k | PASS; first confirmed update 375 | PASS; first confirmed update 375 | [Control GIF](media/control-gaussian1d_smoke.gif), [candidate GIF](media/candidate-gaussian1d_smoke.gif) |
| Own Gaussian stability, through6k | FAIL | FAIL | [Control GIF](media/control-gaussian1d_stability.gif), [candidate GIF](media/candidate-gaussian1d_stability.gif) |
| Broad vector guardrail, 1.2k | PASS; suffix 22 | PASS; suffix 22 | [Control GIF](media/control-vector_two_broad.gif), [candidate GIF](media/candidate-vector_two_broad.gif) |
| Native grid100, 7k | FAIL; 0/5 terminal passes | FAIL; 0/5 terminal passes | [Control GIF](media/control-grid100.gif), [candidate GIF](media/candidate-grid100.gif) |
| Fixed two-pole control, 80 | PASS; separate zero/stored-weight fixture | BLOCKED; scalar replay is unsupported on this component host | [Control GIF](media/control-two_pole.gif) |

The control has **3 PASS / 2 FAIL**, and the candidate **2 PASS / 2 FAIL / 1 BLOCKED** in this five-task diagnostic scope. All nine runnable jobs completed, with zero scientific retries, incomplete workers or invalid receipts. The fixed two-pole result has its original separate identity; its candidate blocker spends zero. Smoke permits an earlier independently confirmed state while completing1k; neither arm's terminal smoke sample passes all location/width/CDF bounds. It must not be described as sustained endpoint quality.

The trainer delta is only public `Recipe.finite_step_mode='armijo'`. Each actual joint G/prior proposal uses the winner's optimizer direction and D update, then tests

`L(theta + a*delta) <= L(theta) + .1*a*(gradient dot delta)`

at a=1,1/2,...,1/1024 on the same batch and frozen post-D critic. Acceptance additionally requires strict decrease for a negative slope. A failed finite search takes zero parameter motion. Optimizer history advances once per proposal. The replay retains drawn latent rows/kernel offsets and restores model buffers/stochastic RNG around extra forwards. It reads no target centers, widths, labels or evaluation samples.

[Armijo's original analysis](https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-p.pdf) motivates finite sufficient decrease. [Stochastic line-search research](https://arxiv.org/abs/1905.09997) studies same-batch acceptance under interpolation assumptions. Those convergence rates are not established for this changing GAN game and normalized direction. The finite inequality is the implementation property; fidelity is the empirical question.

The [original-source reversible probe](saved-probe.json) restores source 753f28a5 and the complete archived seed0 grid100 endpoint. Its actual next D/G proposal has first-order batch-loss change **-.129034** but finite full-step change only **-.003762**. The full step decreases loss yet fails sufficient decrease, with median finite output travel **7.607 target sigmas** and diagnostic precision **.23970 → .23435**. Half scale gives **.61550** diagnostic precision. This is one discarded next-step reconstruction with oracle geometry, not a continuation, served-law gate or actual-trained candidate. All parameter, optimizer, buffer and named/global RNG identities are restored. Failed disposable preparation probes and software checks are disclosed in [diagnostics](software-diagnostics.json).

Across the trained native candidate's **7,000** updates, every actual proposal is locally downhill. **6,884** full proposals nevertheless increase the finite batch loss; another **two** decrease loss but fail sufficient decrease. **6,886** proposals are shrunk, **16,034** halving reductions are used, and all **7,000** accept finite decrease with zero rejection. Mean accepted scale is **.226304**, median **1/4**, minimum **1/8**. An independent arithmetic audit of every raw proposal finds zero Armijo violations. Thus a trigger that checks only first-order conflicts would miss the native overshoot measured here. This is not a distribution-fidelity guarantee.

| Independent 100k live holdout metric | Control | Candidate | Frozen requirement / interpretation |
| --- | ---: | ---: | --- |
| Precision | .24072 | .65938 | >=.97 |
| MassTV | .14718 | .14892 | <=.10 coverage and <=.06 accuracy |
| Modes with genuine quality mass>=.005 | 8 | 63 | 100 required; nearest-cell occupancy is100 in both |
| Minimum genuine quality mass | .000020 | .001740 | >=.005 in every mode |
| In-radius center RMS, target sigmas | 1.52031 | .41187 | <=.20 |
| In-radius absolute covariance trace bias | .37138 | .71006 | <=.10 |
| In-radius radial KS | .49088 | .24821 | <=.04 |
| Full-cell covariance Frobenius RMS | 62.9694 | 87.2746 | Uncensored diagnostic; worsens |
| Full-cell mean covariance trace / target trace | 10.7909 | 18.4033 | Uncensored diagnostic; worsens |
| Full-cell minimum / maximum covariance eigen ratio | .4336 / 579.270 | .1510 / 399.542 | Diagnostic, including spill |
| Full-cell center RMS, target sigmas | 4.28907 | 1.54963 | Diagnostic, including spill |
| Full-cell radial KS mean | .82468 | .46691 | Diagnostic, including spill |

The full-cell statistics include every nearest-cell sample, including spill. They are conditional on evaluation geometry, not identified latent mixture components. Candidate spill falls **.75928 → .34062**, yet covariance Frobenius RMS and average trace worsen. The core covariance and uncensored covariance both need attention: higher precision is not a width repair. Undefined low-count scheduled accuracy fields remain null; the 100k holdout supplies enough evidence for these separate summaries. Both native arms fail all five final scheduled checks and the independent holdout. The candidate prediction is falsified by precision below .97.

Gaussian's final shifted-target KS improves **.320623 → .077984**, but remains above .05; terminal mean error/std ratio become **.101608/.829855**. The full stationary checks pass only **1/72** for the candidate versus **2/72** for control, shifted hold **1/24** versus **0/24**, and deadline reacquisition fails for both. Broad guardrail metrics are exactly equal: SW1=.134590, HQ=.988770, MassTV=.062988, covariance error=.385581, minimum eigen ratio=.400066 and passing suffix 22. The global finite rule changes no accepted broad proposal. These results do not supply a new winner across tasks.

Both arms were trained from the same frozen source [6dca6707](https://github.com/255BITS/ParticleGAN/commit/6dca6707c404417fec72f29877840225915f5e9f), execution digest `6e9a37a4f370135151fae4e30a02dd64ef565fbff5ada39e6b45a768a682514d`, with seed0 and the public deterministic initializer. One complete winner recipe is used globally: nonsaturating/full DualNorm .001 smoothing/momentum 0/per-offset, G .012/D .018/prior .030, constant floors 1, BCAP cap 1 / coefficient 1 / every update, prior_reg 0, no additive training output noise or EMA. Each task retains architecture, target/data law, prior/sigma/weights, sampling, updates and scoring cadence. Vector/Gaussian adapters reuse one real tensor for D/G as frozen. Constructor/data/prior/kernel/model/evaluation streams are isolated and checkpointed. Own Gaussian smoke states retain their original prefix/schedule.

[Protocol proofs](provenance.json) verify equal initial models and named training-stream states. Scalar Gaussian receipts retain equal actual batch-sequence digests. Broad/native adapters retained final data RNG states rather than tensor traces; a separate [deterministic reconstruction](batch-sequence-replay.json) hashes every frozen data batch and exactly reproduces both arms' consumed states. No trained RNG is mutated. [Inactive parity](inactive-parity.json) confirms the control's complete native model and optimizer tensors exactly reproduce the archived winner. [Original receipt certificates](receipts.json), [final metrics](results.json), [scorer controls](scorer-controls.json) and [nine media receipts](media/index.json) preserve evidence identity. Target-law oracle samples pass; point-collapse/one-mode controls fail. GIF export adds zero updates or sample draws.

Training charged **1,199.949529 worker seconds**, against declared full reservations **12,840 seconds** (executed full allowances 12,540; unsupported two-pole uses none), inside the track ceiling 14,400. Separate preparation/software/scorer and data-reconstruction reservations add **360 seconds**, for a combined ceiling 13,200. The completed original probe costs 2.133 seconds and data reconstruction 3.599 seconds; failed disposable preparation costs have their explicit reservation and are not invented as exact measurements. There are no scientific retries or live reservations. Contended wall time is accounting, not an optimizer-speed comparison. Raw stdout, JSONL, checkpoints and state dumps stay on the artifact drive.

The publication adds an **UNMEASURED API guard** refusing standardized priors: fixed-row replay does not represent their global reparameterization. Every trained task uses standardize=false, and measured source 6dca6707 remains authoritative. This software-only refusal and the publication/coordinator corrections are not a new trained arm. Validation passed 109 trainer/protocol/parity checks plus 75 scoped checks after the guard, and Forge validation; summaries-only memory refresh preserves original qualification and telemetry snapshots.

Recommendation: stop this exact finite-step package and retain the exact winner as the matched research reference. The useful finding is actual downhill finite overshoot, now measured rather than inferred from cubature. The uncorrected density/tail covariance and mass imbalance are the remaining question; first isolate their saved-state contribution with a separately bounded proposal before changing another global mechanism. No seed experiment, sweep, unchanged continuation, automatic second arm, pooled qualification or public-default promotion follows.

[PR370](https://github.com/255BITS/ParticleGAN/pull/370) stays unmerged. Reproduction sources are [prepare.py](prepare.py), [run.py](run.py), [probe_saved.py](probe_saved.py), [publish.py](publish.py), [scorer_controls.py](scorer_controls.py) and [replay_batches.py](replay_batches.py); completed runs are evidence, not permission to rerun unchanged science.

```sh
# Saved execution streams, retained after the bounded drain exits:
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/native_overshoot/logs/driver.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/native_overshoot/queue/events.jsonl
# Re-export saved observations; constructs no models and adds no updates:
PYTHONPATH=$PWD /home/martyn/dev/ParticleGAN/.venv/bin/python \
  reports/forge/bcap-physics/native_overshoot/round4/publish.py
```
