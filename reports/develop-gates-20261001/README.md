# Develop verification and gate diagnosis

The integrated Atlas formulation passes all **19/19 original CUDA gates**.
The [independent replay receipt](atlas19-replay.json) binds the original hosts,
constructors, scorers, budgets, observation schedules and noisy serving law.
All three native 7k endpoints match the original qualified final metrics
exactly. No `particlegan/` or Atlas configuration file changed during this repair.

[Software CI](https://github.com/255BITS/ParticleGAN/actions/runs/36897240410)
passed **2,249 tests, 69 skips and 18 subtests**, plus the installed-wheel
smokes and distribution build. The [software receipt](software-ci.json) also
records 37 passing history/memory checks for this report's catalog registration.
The release branch and tags remain unchanged.

The [complete portable CPU legacy BCap run](common22-ci-portable-runtime.json)
remains **19/22 FAIL**: all three native gates pass, but trajectory, mode hold
and unequal-width vectors fail. The sampling-law mismatch is repaired; these
are separate scientific failures under the retained legacy gates. The
clean-MoG controls also remain failed. This is not an all-cohort release pass.

| Original Atlas gate group | Fresh valid passes | Original execution budget |
| --- | ---: | --- |
| Native coverage, final-five fidelity and independent 100k holdouts | 3/3 | 7,000 updates each |
| Mode hold, four images, six vectors, stationary and ring shift | 13/13 | Frozen per-host budgets; stationary 7,500 and shift 4,600 |
| Moving grid, rotated and staggered targets | 3/3 | 1,500 updates each; two 30-degree turns |

The replay executed 48,800 updates in 6,546.91 wall seconds, including
evaluation, diagnostics and current contention. This is not a throughput
claim. The audit checks every observation schedule, strict RNG transactions,
the native construction/scorer receipts, complete checkpoints, backend
selection and optimizer-role calibration. It independently regrades native
clouds and portability/moving observations. All three static native clean
diagnostics remain **FAIL**; neither these results nor the original noisy
passes supply current Forge MoG/clean-live qualification.

## Why the develop CI job failed

The failed develop job did not test E22 or Atlas. It ran the legacy BCap
configuration through the historical 22-host runner. PR #209 had changed the
three native problems to clean scoring, while the 19 transfer hosts retained
their original output-noise sampling law. The combined checker compared the
training noise settings but omitted this evaluation-law difference.

The native clean failures are real under that law. Their within-mode covariance
is about 98% below the target: the generator's narrow points are insufficient
without the .029 output-noise component used in training and in the original
served qualification. These failures do not establish an E22/Atlas regression.

## Paired saved-draw evidence

The [diagnostic receipt](paired-sampling.json) binds the exact original sample
files from [Actions run 36829031177](https://github.com/255BITS/ParticleGAN/actions/runs/36829031177).
It adds only the declared .029 output noise using the existing evaluation seeds.
No model was retrained and no original receipt or acceptance threshold changed.

| Native problem | Clean terminal passes | Original noisy law terminal passes | Clean holdout covariance bias | Noisy holdout covariance bias | Noisy holdout radial KS |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid100 | 0/5 | 5/5 | -.98040 | -.04194 | .01713 |
| rotated100 | 0/5 | 5/5 | -.97536 | -.03826 | .01594 |
| staggered100 | 0/5 | 5/5 | -.97836 | -.03974 | .01621 |

All three 100k-draw noisy holdouts pass the unchanged coverage/fidelity limits.
This is an explanatory paired-draw diagnostic, not a new training qualification.
The repaired checker grades the original mixed-law aggregate **INCOMPLETE**,
while preserving all three native clean failures and transfer **19/19 PASS**.

## Repair and verification

Native toy100 scoring remains clean by default. `--eval-output-noise` explicitly
selects the benchmark wrapper's original served law for both terminal and
holdout draws. The common-22 runner now selects that law before training,
records it, checks it against the transfer law, and rejects clean-only requests
before spending. The CI workflow names the actual legacy BCap cohort and binds
its configuration explicitly. It supplies no clean-sampler or Atlas credit.

New software probes compare complete trainer checkpoints across clean/served
scoring, including models, optimizers and RNG streams, and check the independent
holdout writer. Additional negative controls reject mismatched sampling laws
even when every metric otherwise passes. No `particlegan/` file is changed.

The restored common-22 runner passed **22/22** in the
[first local runtime](common22-local-original-runtime.json). Its next hosted
execution measured **19/22 FAIL**: grid100 and two unequal-component vectors
failed, while the other 19 cases passed. The
[failed hosted receipt](common22-ci-original-runtime.json) remains separate.
Those are true failures under that runtime, not Atlas failures.

The old hosted jobs capped CPU instruction sets without fixing a common BLAS
arithmetic branch. Native initial samples match exactly, but paired draws
diverge at rounding scale on update 1 and amplify. Two bounded software probes
using the same CPU wheel, full native resource sizes and unchanged 7k schedule
verify exact checkpoint and RNG equality across clean/served scoring after
40 updates: [original dispatch](cpu-sampling-probe.json) and
[portable branch](cpu-sampling-probe-compatible.json). They cost 160 updates
in total and earn no scientific quality credit.

The workflow now declares portable CPU BLAS profile v2, records its CPU/build
identity and sets `MKL_CBWR=COMPATIBLE`. Its complete
[Actions run 36897240453](https://github.com/255BITS/ParticleGAN/actions/runs/36897240453)
measured **19/22 FAIL**: native **3/3 PASS**, transfer **16/19 PASS**.
Independent grading verifies all 140 captured training-source files against
the executed commit, all sampling/source/recipe identities, all native final-five
and 100k holdout gates, and each retained transfer verdict. All three native
terminal windows pass 5/5. The hosted CPU is an AMD EPYC 7763; its uploaded
archive is preserved unchanged and matches GitHub's SHA256 digest.

| Remaining legacy failure | Full update budget | Final metric | Retained requirement |
| --- | ---: | ---: | --- |
| trajectory | 400 | identity MSE .250462 | <= .02 |
| mode_hold | 1,200 | 5 modes, HQ 1.0 | >= 8 modes and HQ >= .9 |
| vector_unequal_width | 1,200 | minimum component eigen ratio .114356 | >= .15 |

All three have a zero passing suffix. Unequal-width vectors passed seven earlier
observations, then failed the terminal shape requirement. The grader agrees with
the stored host verdicts; neither an earlier window nor EMA diagnostics can
replace the declared live result. No execution error or missing budget explains
these failures. The original Atlas mode-hold and vector gates remain passing
under their separately bound original protocol; no cross-cohort credit is added.

The [first local portable attempt](common22-portable-local-timeout.json) hit its
registered 40-minute execution cap after two complete native passes and a partial
third. The [first hosted attempt](common22-portable-ci-timeout.json) also timed
out at 45 minutes: two native passes, a partial third and no transfer execution.
GitHub's timeout annotation is retained; neither attempt earns aggregate credit.
Every saved draw from their two completed tasks matches exactly across the local
Ryzen 5900X and hosted EPYC 7763: [82 NPZ files and 246 arrays](cpu-portable-saved-draw-parity.json),
with no new training. This establishes only the observed two-task parity, not
an explanation of the earlier hosted hardware whose identity was not recorded.
The measured cost prompted the completed 75-minute CI retry, without altering
training-update budgets, numeric settings or acceptance limits. The study is
concluded with these failures preserved; no further recipe, numeric-profile,
seed or continuation search follows. See [the frozen protocol and timed-execution
amendment](PROTOCOL.md).

The manifest repair required an explicit
[evaluator revision](evaluator-revision.json) in 14 Forge task declarations.
Only the evaluator source identity and revision metadata changed; execution,
sampling, thresholds and budgets remain the same. Old receipts keep their
original fingerprints and receive no implicit new qualification.

Raw Atlas logs, checkpoints, clouds and all external reproduction sources
are archived under `/ml2/hypergan/gan-attempts/develop-gates-20261001/atlas-original19/`.
Its inventory hash binds all 558 files (163,735,386 bytes). Other raw CPU
artifacts are preserved in receipt-linked directories and original GitHub ZIP
archives under `/ml2/hypergan/gan-attempts/develop-gates-20261001/`.
Only compact reports, final metrics, provenance and reproduction sources are
tracked, as required by `AGENTS.md`.

This resolves the formulation-preservation question and the original Atlas
qualification. A release still needs its declared serving/prior scope:
current Forge MoG/clean-live adoption remains blocked, with its earlier
failed controls preserved. No default promotion or release follows this study.

The concern that K3P alone did not solve the current Forge cohort remains
supported by the [completed formulation comparison](../forge/FORMULATION_COMPARISON_READOUT.md).
Its K3P control reached 100 modes but passed none of the five terminal joint
fidelity checks. Adding the same scheduled output noise over-widened that MoG
cohort and still failed; the sampling-law repair here does not rescue those
controls. Fixed BCap, R1/R2 and both release adaptations also failed their full
7k gates. Those failures require a separately justified formulation/task study,
not relabelling with this original cloud/noisy Atlas result.
