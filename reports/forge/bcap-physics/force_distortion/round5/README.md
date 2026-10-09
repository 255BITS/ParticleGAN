# Force distortion, round five

Damped output-kernel filtering creates **no new full PASS**. Both matched arms finish **4 PASS / 2 FAIL**, with no blocked final cells. Native holdout precision improves from .24072 to .28345, below the preregistered .48 forecast; full native coverage/accuracy and Gaussian retention fail. Conditional identity passes survive. **Stop this exact global filter; retain the direction blend for its measured conditional repair, without treating it as a complete distribution or retention repair.** This is a bounded research diagnostic, not ordinary qualification or default adoption.

## Matched numerical results

All twelve final jobs complete their original update budgets. Numerical certificates determine these grades; GIFs illustrate actual saved training states. [Results](results.json), [compact receipts](receipts.json), [provenance](provenance.json), [validation](validation.json) and [media provenance](media/index.json) retain the full metrics, sustained checks and source bindings.

Forge concludes both [control](../../../records/readout-b58a2b1466f992da0e0096fc.json) and [candidate](../../../records/readout-e4aa4ff1deff6bbebdbd4df6.json) studies. The candidate's declared decision outcome is `inconclusive` because neither its prediction nor explicit falsifier is observed; its next action is `stop_and_readout`, and no further execution is authorized. That administrative outcome does not change the two complete gate failures.

| Task / total updates | Direction-blend control | Output-filter candidate | Actual-training GIFs |
|---|---|---|---|
| grid100 / 7,000 | FAIL; precision .24072, mass TV .14718 | FAIL; precision .28345, mass TV .14981 | [control](media/control-grid100.gif), [candidate](media/candidate-grid100.gif) |
| Gaussian smoke / 1,000 | PASS; first confirmed step 375 | PASS; first confirmed step 584 | [control](media/control-gaussian1d_smoke.gif), [candidate](media/candidate-gaussian1d_smoke.gif) |
| Gaussian stability / 6,000 including own smoke | FAIL; stationary 2/72, shifted hold 0/24 | FAIL; stationary 4/72, shifted hold 1/24 | [control](media/control-gaussian1d_stability.gif), [candidate](media/candidate-gaussian1d_stability.gif) |
| trajectory / 400 | PASS; identity MSE .000258480, passing suffix 19 | PASS; identity MSE .000235034, passing suffix 16 | [control](media/control-trajectory.gif), [candidate](media/candidate-trajectory.gif) |
| residual student / 400 | PASS; identity MSE .000249089, passing suffix 21 | PASS; identity MSE .000237168, passing suffix 19 | [control](media/control-residual_student.gif), [candidate](media/candidate-residual_student.gif) |
| mid-scale identity / 800 | PASS; identity at 0 / mid .984797 / .992997, suffix 20 | PASS; identity at 0 / mid .990636 / .976050, suffix 22 | [control](media/control-mid_scale_identity.gif), [candidate](media/candidate-mid_scale_identity.gif) |

Native figures use the independent 100,000-sample clean/live holdout. Both fail all five terminal accuracy checks and the coverage gate. All 100 nearest-cell regions contain some uncensored samples, but only 8 control / 16 candidate modes have genuine radius-qualified mass >= .005; minimum genuine mass is .00002 / 0. The candidate's zero genuine mass leaves its coverage-conditioned center/covariance/radial statistics undefined. These nulls remain null, not favorable zeros. Uncensored covariance Frobenius RMS improves 62.9694 → 56.2847 while spill remains 75.928% → 71.655%. This is a modest precision/shape improvement with severe unresolved density and mass errors, not recovered coverage. The target holdout passes both scoring families.

Gaussian stability restores each arm's own certified smoke prefix exactly, then completes its original stationary and shifted phases. Both fail shifted reacquisition and all 48 frozen observations. Final shifted KS is .320623 / .219856, and std/target-std is .662330 / .776934. A smoke pass or slightly better endpoint cannot replace failed sustained retention. The two initial control-smoke executions timed out during independent grading after raw 1,000-step completion; their original **INCOMPLETE** receipts and costs remain in the retry history. Two same-source execution retries produced the final certified smoke PASS. No completed quality failure was retried.

Both residual arms have success rate 1 and wrong-pad rate 0. The candidate's slightly lower conditional endpoint MSE comes with shorter trajectory/residual passing suffixes; one fixed protocol does not establish statistical superiority. Its mid-scale identity score at .5 declines but remains above .85, and concept cosines/magnitudes pass their original bounds. The explicit trajectory MSE > .02 falsifier is not observed; the native >= .48 prediction is not observed, and the complete global-repair claim fails.

## Evidence and intervention

The [saved force traces](saved-force-traces.json) precede mechanism selection and retain original attempt/checkpoint identities. Native uses the original winner's frozen source and one discarded actual next post-D G proposal, restoring parameters, optimizer/buffers and named/global RNG state. Conditional probes fix the saved critic and retain the original objectives; unequal-width vector cubature is a separately labelled deterministic cohort. These are endpoint probes, not a causal decomposition of training history. Native motion RMS uses per-coordinate RMS; it should not be silently pooled with another report's row-vector RMS.

Native raw shared-gradient motion sends 5.08% of rows uphill against their own output force, versus 3.61% after FullDualNorm and 3.76% for its finite move. The vector figures are 30.18%, 21.19% and 21.29%. Finite motion remains close to its linear approximation, with relative errors 2.49% native and 2.21% vector. Network/prior linear contributions add with relative error below 1.3e-7. Conditional individually normalized proposals have nonadditivity .996914 / 1.215825, reproducing the original endpoint finding. Their self-objective uphill fractions are zero in these probes; that does not certify conditional identity throughout training. Aggregate descent does not protect every output row, and FullDualNorm can improve some measured alignment rather than invariably worsen it.

The sole candidate filters the unchanged objective's sample-output force:

```
f = d loss / d actual sample outputs
J = joint generator + learned-prior output Jacobian
lambda = ||J^T f||^2 / ||f||^2
u ≈ lambda (J J^T + lambda I)^-1 f
parameter gradient += J^T(u - f)
```

The inverse uses at most four matrix-free CG iterations with fixed numerical early convergence. Zero-force cases use identity handling. Only the output pullback changes; direct latent regularizers remain intact. The original FullDualNorm proposal, rates, critic optimizer and direction blend then execute. No target labels, evaluator geometry, component sigma or extra supervision enters this rule. This is a damped empirical output metric, not a Fisher construction or finite acceptance guarantee.

Filtering is active on every candidate committed G update. Candidate checkpoints record 7,000 native, 6,000 continued Gaussian, 400 trajectory, 400 residual and 800 mid-scale filtered updates; smoke is the reused 1,000-step prefix, not additional stability updates. Maximum relative CG residual is .9900 native, 2.4554 Gaussian, .1195 trajectory, .1570 residual and .0706 mid-scale. Four iterations are therefore not an exact solve or uniform residual guarantee. The negative conclusion binds this approximation, damping law and downstream optimizer; it does not reject every possible function-space method.

[Jacot, Gabriel and Hongler](https://proceedings.neurips.cc/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf) motivate tracing empirical J J^T coupling. Their infinite-width gradient-flow convergence results do not certify this finite adversarial trainer. [Amari](https://doi.org/10.1162/089976698300017746) motivates distinguishing parameter and distribution geometry; the construction here is an analogy rather than his Fisher natural gradient. [Martens and Sutskever](https://www.cs.toronto.edu/~jmartens/docs/HF_book_chapter.pdf) provide primary matrix-free CG/preconditioning context. Neither this analogy nor batch descent proves mode coverage or retention.

## Protocol, scope and preservation

The control is the exact retained round4 direction blend, selected because it already repaired both measured identity tasks. Both arms run on **one frozen scientific source**, committed and pushed before training:

- Source commit: `1a977eec9838918f3722921c49fe2f4b632fe3e4`; execution digest: `d289a2c2d679513f53b49768068aa61e3f8956292244028d074d1d997844e54c`.
- Control: `force-distortion-round5-control-v1`, request `5829b31118fae0e93f3fc4a4`; candidate: `force-distortion-round5-candidate-v1`, request `a5b26912f3a1e4c298758378`.
- Ready [control study](../../../../../configs/forge/studies/force-distortion-round5-control-study-v1.json) and [candidate study](../../../../../configs/forge/studies/force-distortion-round5-candidate-study-v1.json); private campaign `force-distortion-round5-v1` and research view `force-distortion-round5-diagnostic-v1`.
- Sole recipe delta: `constraint_geometry_mode: direction_blend → sample_force`. FullDualNorm smoothing .001, momentum 0, per-offset convolution, G/E .012, D .018, prior .030, floors 1, no additive training output noise or EMA remain common.

Seed 0 and the public deterministic initializer are common. All six pairs have identical initial generator/critic/prior hashes and every consumed non-evaluation stream hash. Constructor, data/batches, training noise and evaluation streams remain separate and checkpointed. Actual Gaussian data hashes also match; native batch draws follow the same frozen source and matching consumed data streams. Both arms preserve each task's architecture, target/data law, prior, sampling, update budget and evaluation cadence. All 258 recorded RNG audits show zero unintended deviations. Scalar/native adapters retain their same-real-tensor D/G law. Original conditional cloud exceptions remain explicit; no fixed identity/zero fixture substitutes for a shared learned-MoG baseline.

GANTrainer and trajectory/residual bind their existing generated tensors through the public helper. Mid-scale binds existing repeated adversarial outputs and existing cover outputs, in original order: it adds no forward, loss or supervision, but the declared output metric counts those objective branches separately. Unsupported component hosts fail preflight. Control consumers are inactive. Native training remains adversarial-only with prior regularization 0; no training identity/set-coverage objective has been invented.

The diagnostic denominator is these six tasks only. Two-arm initial full reservations are 19,440 seconds; two 120-second execution retry reservations raise this to 19,680, leaving 1,920 seconds for ancillary checks/probes within the **21,600-second track ceiling**. Certified worker accounting is **3,832.324 seconds including the two incomplete attempts**; one timeout's measured process duration exceeds its reservation, and its full original charge is retained. Ancillary work is separately bounded by its remaining allowance, not represented as trained qualification. The published successful saved-state trace took 1.663 seconds with zero committed updates; synthetic/software checks are separate cohorts. Rare and broad full reservations do not fit and were not run. GPU contention is accounting, not a speed comparison.

The [read-only audit](validate_saved.py) verifies 523 scientific files against the executed source, all original task conditions, 14 attempt identities, own-prefix restoration, streams and 12 actual-training GIFs. Software checks pass **225 unique tests** (224 broader tests plus one new public nonlinear GANTrainer test; the final four-test suite overlaps three). Independent [scorer controls](scorer-controls.json) accept the numerical target oracles and reject point collapse, one-mode mass, wrong Gaussian mean, wrong conditional identity and mid-scale identity swaps. These are target-informed evaluation witnesses only. Conditional GIF rendering strips structured metrics from display without changing scoring.

Publication changes only reports/reproduction helpers. Eight qualification/telemetry snapshots present at the stacked base remain byte-for-byte unchanged; six ignored JSON board snapshots absent there remain absent. Older evidence stays under its original source/initialization/cohort. Compact results, receipts and media are committed; raw stdout, JSONL, tensor states and checkpoints remain in the artifact archive. The parent owns the one current goal leaderboard; this table is the scoped study comparison. Refresh memory only with `compile --summaries-only`, then check it. Screening profiles remain provisional.

## Reproduction and next action

[execute.py](execute.py) uses the public Queue/drain API with completion compilation disabled and both A6000 GPUs shareable. Raw archive: `/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/force_distortion`. To inspect existing execution logs:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/force_distortion/logs/driver.log
```

To reproduce the saved readout without training, use the repository Python 3.12 environment and this worktree's PYTHONPATH, with the receipt/archive paths above available:

```sh
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/force_distortion/round5/publish.py
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/force_distortion/round5/scorer_controls.py
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/force_distortion/round5/validate_saved.py
```

[probe_saved.py](probe_saved.py) reproduces the separately scoped original-source endpoint probes; it commits no updates. Running the two ready arms again would be a new paid experiment, not a readout step. Stop this exact filter after the bounded comparison. Preserve the measured conditional direction-blend repair and use the parent's aggregate round5 evidence before admitting another intervention; no sweep, seed repeat, continuation, extra task or automatic promotion follows.
