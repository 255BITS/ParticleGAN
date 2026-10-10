# BCAP develop integration and full suite comparison

The direction blend and local-v2 transport research is implemented as opt-in capabilities across the full Tier 1/Tier 2 host suite in [PR377](https://github.com/255BITS/ParticleGAN/pull/377), based on develop `5737ade47dca89b04d338ada20667781d1f2b5df`. The incumbent reproduces **6/6 Tier 1 and 7/21 Tier 2**. The combination repairs trajectory, residual student, and unequal mass in separate Tier 2 diagnostics, but regresses two-pole and times out on words in Tier 1. **Keep this PR in draft and retain the incumbent; the proposed opt-in merge bar has not passed.**

The [task table](task-table.md), [metrics and provenance](results.json), [aggregate failure analysis](failure-analysis.json), [evidence audit](audit.json), and [saved-training media index](media/index.json) retain ordinary and diagnostic results separately.

| Arm and evidence scope | Tier 1 | Original Tier 2 questions |
| --- | --- | --- |
| Incumbent, ordinary | 6 PASS / 6 | 7 PASS / 21; 14 FAIL |
| Combined, ordinary | 4 PASS, 1 FAIL, 1 INCOMPLETE / 6 | All 21 BLOCKED by the Tier 1 screen |
| Combined, research diagnostic | Own Gaussian producer PASS; other Tier 1 questions outside this lane | 9 PASS / 20 measured; 11 FAIL; word hold BLOCKED |

On the **same 20 measured Tier 2 questions**, the comparison is **6/20 incumbent passes versus 9/20 combined diagnostic passes**: three repairs, six retained passes, eleven retained failures. The incumbent's seventh pass is word hold, which cannot be measured for the combination. Diagnostic gains do not establish full-suite qualification.

## What was integrated

Both arms retain the original winning BCAP recipe: non-saturating adversarial loss, per-offset FullDualNorm, smoothing `0.001`, constant generator/encoder step size `0.012`, discriminator `0.018`, learned prior `0.03`, and discriminator gradient soft cap `kappa=1`, coefficient `1`, every update. The [incumbent configuration](../../../configs/forge/ideas/bcap-develop-integration-winner-v1.json) and [combined configuration](../../../configs/forge/ideas/bcap-develop-integration-combined-v1.json) retain the full explicit settings. The combined arm has exactly this global delta:

```python
constraint_geometry_mode = "direction_blend"
kinetic_transport_weight = 1.0
kinetic_transport_local_weight = 1.0
kinetic_transport_projections = 32
```

Direction blend searches for a common descent direction for the host's existing protected generator objectives, including prior parameters owned by the same optimizer. It checks the actual rounded displacement's first-order derivatives. It retains the normalized step magnitude and does not perform a finite-loss line search. The global transport objective matches empirical projected quantiles in 32 deterministic directions, normalized by detached real coordinate variance. Local-v2 compares relative kernel moments at detached real batch anchors, using fourth-other-neighbor radii at multipliers 1, 2, and 4. These finite objectives do not establish full Wasserstein convergence, population density recovery, or monotonic adversarial losses.

Custom hosts use their original generated output panels and real batches. Context, labels, and latent coordinates do not enter output distance; the unused-token hold excludes the held row. AE transport uses existing unconditional decoder outputs. Trajectory, residual, unipolar, cover, midscale, and word hosts retain their original supervision and forwards. The word distance uses flattened categorical probabilities against real one-hot batches. Native, vector, image, ring, and mode-hold tasks use the shared trainer. Enabled mechanisms have explicit consumer contracts and checkpointed activation counters.

Public defaults disable both features. Other techniques retain their losses, routing, optimizer state, clocks, priors, and consumed streams; they need no projection or transport hooks. Historical Forge `bcap` remains Adam-backed, separately identified as `bcap_adam`; the comparison explicitly requests the winning DualNorm recipe.

## Comparison contract and eligibility

The [preregistration](preregistration.json) specifies one global configuration per arm, protocol seed 0, public deterministic initialization, identical task architectures, target/data laws, actual batches, priors, sampling laws, update budgets, and evaluation cadence. Constructor, data, training-noise, and evaluation streams are separate and checkpointed. The [original task archive](original-task-contracts.json) preserves all 27 original numerical questions and gates from develop. Source and optional consumer declarations are versioned explicitly. Original fixed fixtures, including midscale's zero KEEP fixture, remain identified in their original cohorts. Clean and scheduled-noise results are separate task cohorts.

Ordinary eligibility requires all six Tier 1 tasks to pass before Tier 2. The combined arm fails that screen. Its [separate admitted diagnostic study](deeper-preregistration.json) therefore measures only the 20 Tier 2 tasks with eligible checkpoint dependencies, plus its own full-budget passing Gaussian prefix producer. It cannot borrow the incumbent's producer or fill ordinary qualification cells. Its word hold stays blocked because its own word producer is incomplete. Neither lane feeds historical qualification or changes the current technique inventory.

## What improved

The three matched FAIL-to-PASS repairs are:

| Task | Incumbent final metric | Combined diagnostic final metric | Sustained evidence |
| --- | --- | --- | --- |
| Trajectory | Identity MSE `0.239862` | `0.000236766` | 18/24 passing checks, suffix 18; confirmation at update 184/400 |
| Residual student | MSE `0.061036`, success `0.5`, wrong pad `0.5` | MSE `0.000222020`, success `1`, wrong pad `0` | 20/24 passing checks, suffix 20; confirmation at 150/400 |
| Unequal mass | Full component covariance error `3.69165`, minimum mass ratio `0.20752` | `0.38667` and `0.87703` | 18/24 passing checks, suffix 9; confirmation at 1000/1200 |

Midscale, broad components, spiral, unipolar, cover, and stripes were already incumbent passes and remain successes. They are not additional repairs. The effects of direction blend and transport have not been isolated by this combined comparison.

## Remaining failure modes

**Finite-step stability and retention.** Gaussian retention still fails the stationary hold, deadline reacquisition, and shifted hold. The combined endpoint passes with KS `0.03367`, standardized mean error `0.000224`, and standard-deviation ratio `1.1667`, but only 6/72 stationary and 2/24 shifted checks pass; frozen retention is 0/48. A good final distribution does not satisfy continuous retention. Overlap and blobs likewise reach passing endpoints but have terminal suffixes of only 2 and 1. Anisotropic quality falls from `0.9607` at update 1150 to `0.2112` at 1200, while maximum component spill rises to `0.9105`.

**Full density and tails.** Unequal width passes sliced distance and mass TV at all 24 checks while failing full component covariance at all 24. Its final full covariance error is `3.5905`, above the `0.85` gate, although core covariance error is `0.1714`. Global mass and core metrics hide off-core spread. Bars ends with high-quality fraction 1 but only 3/4 modes. Intensity ends with high-quality fraction 0 and no quality modes despite nearest-template TV `0.03125`. Mode hold reaches eight modes but quality `0.7654`, below `0.9`.

**Native 100-Gaussian quality.** Grid improves holdout precision from `0.24072` to `0.33706` and TV from `0.14718` to `0.08489`; final live quality modes improve from 9 to 22/100. Rotated improves TV from `0.14325` to `0.09120`, but precision falls from `0.25552` to `0.16744`; modes change from 13 to 15/100. Staggered improves TV from `0.12277` to `0.07563` while precision falls from `0.30168` to `0.21988` and quality modes fall from 20 to 12/100. Both arms fail all five terminal accuracy checks on all three tasks. Coverage requires precision at least `0.97` and all 100 quality modes; additional accuracy gates include TV at most `0.06`. Combined shape summaries are unavailable where a component lacks enough quality-radius samples. All native samples remain finite. The original-law oracle passes with precision `0.98914`, so the gates have a passing reference under the same data law.

**Tier 1 regression.** Two-pole passes at its endpoint but has only 6/24 passing checks and suffix 1, versus the incumbent's 17/24 and suffix 17. Its final five discriminator gradient medians are `1.0068, 1.0266, 1.0194, 1.0278, 0.9852` against a cap of 1. Transport runs on all 80 updates; direction blend records zero conflicts, blends, and stalls. This failure is not explained by blocked projection. Changed generated distributions interacting with finite discriminator steps and its soft cap are plausible; the joint treatment does not identify a unique cause.

**Host scale and execution cost.** The repeated single-concept real panel in unused-token hold has zero coordinate variance. Float32-epsilon normalization allows transport loss to reach `4,194,305`; the host nevertheless passes, with blends on 152/200 steps and no stalls. This exposes a scale sensitivity, not a demonstrated quality failure. Words reaches only update 19168/20001 before its unchanged 900-second allowance expires. Early passing observations cannot certify the full run. This is INCOMPLETE, its hold is BLOCKED, and no certified final checkpoint or actual-training GIF is available for that attempt.

All 26 completed combined provenance checkpoints record zero Pareto stalls and zero maximum protected derivative after the update. Their recorded histories contain 3,224 blends; histories include inherited/replayed producer prefixes and are not a paid-update total. Conflicts vary sharply by host: Gaussian blends on 1292/6000 updates, unused-token on 152/200, and all three natives together on only 7/21000. On native tasks the original GAN objective is protected; auxiliary transport contributes to the proposed gradient and is not itself a protected objective. Enforcing the current first-order condition does not establish finite-loss monotonicity, density retention, or stability with a moving critic. Full normalized steps, finite-batch isotropic anchors, scale degeneracy, and limited local anisotropy resolution are hypotheses to discriminate next.

## Compatibility and evidence handling

The original [compatibility receipt](compatibility.json) retains its exact measured source identity and 63 passing CPU checks against actual develop. It includes scalar parity, actual develop checkpoint continuation, component hosts, words, optional fields, missing hooks, and canonical host admission. Preexisting E22/Atlas policy blockers remain explicit. Active CPU/CUDA checkpoint replay checks cover dense and convolutional optimizers. These bounded software checks do not constitute full-budget qualification for other techniques.

The [complete CPU regression receipt](software-verification.json) records **5,835 PASS, four initial failures, 293 skips, one existing expected failure, and 131 passing subtests in 1,146.06 seconds**, with all 6,133 collected cases visited. Test-only follow-up corrections pass **17/17** checks in 19.52 seconds. Two archived-report checks now replay the exact original nine-view publication rather than include this later diagnostic view. The tiny pacing test now restores the environment setting it previously leaked into the following policy control. Exact develop reproduces that leak and the original package-import/Torch failure; their original assertions and source identities are retained in the [fixture receipt](../../../tests/fixtures/develop-software-regressions.json). The metadata listing question is explicitly scoped to the direct-script CLI, which performs no training imports; package import still loads Torch. Production bytes remain identical to the full-suite launch, and 2,449 protected historical publication files remain byte-identical. The full suite was not repeated to replace its initial result. The separate CI renderer passes 3/3 checks, alongside the targeted CPU/CUDA training and resume checks.

A post-execution AE evaluator fix restores compatibility with existing four-argument wrappers by observing an already-executed decoder forward. It adds no forward or random draw. The [two-arm proof](ae-observer-comparison.json) records exact old/new metrics, guard state, models, optimizers, named streams, scored tensors, encoded GIF bytes, and decoded pixels. This changes evaluator/media plumbing and its source binding after paid execution; it is not a new trained cohort. The primary science audit is retained against the original source. The separate [final-tree compatibility receipt](compatibility-post-execution.json) records **64/64 PASS**, no skips/failures, and 42 exact actual develop/candidate state-hash pairs at commit `62428b007f18272ad8250bfe792423e61dbeb00f`. It verifies that the 52-file original compatibility inventory changed only in the AE host and its source-bound task card, while preserving the original receipt bytes.

Paid scientific source digest: `c68be4ae40db26959c1b2876ed33ec044aba9cf08cab84be71ed8a0264a2eb17`. Ordinary execution originated at `beb7b1df56a51b1119053d6b4a03b6acb119e394`; the diagnostic declaration originated at `1e99c67faee3f1ec40d9683fc932e29c5fe11b8e`. Their scientific bytes are identical and their distinct origins remain recorded. [Prior research evidence](prior-evidence.json) retains PR373's original source and narrower measured scope.

The primary audit passed for **56 paid attempts, 1,205 scientific files, all 27 original task contracts, 54 saved-state comparisons across 49 task/scope pairs, and three own-checkpoint dependencies**. It checked initial models and priors, named training bindings, actual completed stream consumption, activation histories, frozen snapshots, and eight protected historical qualification/telemetry files. Incomplete words cannot establish matched complete consumption and are recorded as such. Audit PASS validates evidence; it does not override any quality result.

Raw stdout, event streams, checkpoints, tensors, JUnit, queue state, and exact certificates remain outside Git at `/mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next`; see the [archive receipt](artifact-archive.json). The publication retains final metrics, all paid attempt identities and costs, checkpoint parents, source digests, and **55 actual-training GIFs**, with no export blockers for completed gates. Incomplete words has no certified GIF and remains explicit. Spiral display overlays the declared analytic noiseless centerline on actual saved generated outputs; it does not draw substitute reference samples. GIFs illustrate certified saved training; export adds zero updates, sampling draws, or regrading. Numerical gates determine the results.

## Costs and reproduction

The ordinary campaign was admitted with a 90,000-second ceiling and 86,040-second full reservation across both arms; the separately admitted diagnostics had a 45,000-second ceiling and 40,320-second reservation. Actual paid charges are **4,066.32 seconds ordinary + 2,548.21 seconds diagnostic = 6,614.53 seconds**, across 35 + 21 attempts, with **zero execution retries** and no remaining runnable work. All native tasks retain 7,000 updates, batch size 2,048, 20,000 learned particles, clean MoG prior width `0.025`, and original terminal/holdout scoring. Shared GPUs affect elapsed cost; these timings are not isolated throughput comparisons. Software verification has its separate declared 3,600-second allowance. The initial Forge profile remains provisional; this execution does not establish calibration or default promotion.

The exact 168 certificate files and five terminal metadata files are archived under `certificates/` in the artifact root. Its manifest SHA-256 is `f8789425fdf31965f1e83dad0f4c3cbc0dc83f847a46763ec756098090a5001a`. Original raw artifact paths and CUDA devices for validating the original named-stream states must remain available. The [reproduction helper](reproduce_evidence.py) restores those exact ignored certificates into a fresh checkout at scientific-source-compatible commit `e1c7fcc22086c290868ea47b989a998a86e913e4` and re-exports evidence without training, sampling, or rescoring:

```sh
python reports/forge/bcap-develop-integration/reproduce_evidence.py reproduce \
  --repository "$PWD" \
  --archive /mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next \
  --checkout /mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next/reproduction-source \
  --output /mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next/reproduction-output
```

Use fresh checkout/output paths for another export. Directly auditing the final AE-patched tree against the earlier trained source should reject the source difference; the helper deliberately verifies the original scientific bytes. [prepare.py](prepare.py), [run.py](run.py), and [deeper.py](deeper.py) retain the declared execution workflow. A future trained comparison on changed source requires fresh admission and source binding; it must not replace these receipts.

The [end-to-end reproduction receipt](evidence-reproduction.json) verifies all 1,205 original scientific files, 56 exact certificates, unchanged final metrics/statuses/paid costs, and byte-identical exports for all 55 GIFs. It preserves the raw reproduction receipt and binds the actual audit/export tool hashes to their Git blobs. Reproduction adds zero training updates, sampling draws, or scorer calls.

Logs remain easy to inspect:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next/logs/driver.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next/logs/diagnostic-driver.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next/logs/final-full-tests.log
```

## Recommendations

1. Retain the incumbent and hold the draft PR. Host support and diagnostic gains are useful, but two-pole regression, incomplete words, blocked hold, and remaining Tier 2 failures prevent the proposed merge bar.
2. Isolate the treatment with matched projection-only and transport-only global variants before attributing repairs or regressions. Keep seed 0 and all public initialization, stream, law, budget, cadence, and gate contracts fixed.
3. Investigate finite-step and prior response on Gaussian, anisotropic, overlap, blobs, and two-pole. Compare a declared finite-loss-controlled update against the current full-step direction; preserve full-curve and terminal requirements. Do not rescue failures with a best checkpoint or activate unrelated optimizer policies under DualNorm without an applicable contract.
4. Investigate local density geometry and normalization on unequal width, anisotropic, rare mass, and native 100. Distinguish global-only, local-only, and combined signals under declared global settings, and test repeated-target/anisotropic behavior as explicit separate fixtures. Balanced assignments and core-only metrics cannot substitute for the original quality gates.
5. Profile word overhead from existing logs and software benchmarks. Any execution optimization or budget change needs its own declared comparison; the existing timeout and blocked hold retain their evidence identities.

The [next Tier 1 repair plan](tier1-next-plan.md) separates the two-pole stability regression from word execution cost, then requires a complete six-task check of one global recipe. It is planned for execution after compaction; its budgets and studies remain draft.
