# Independent CB64-RA validation

**Canonical GPU acceptance: unavailable.** CUDA is inaccessible in the managed sandbox. All new training and scorer evidence below uses the pre-outcome CPU amendment. Package/config, scorer bytes, full budgets and gates are unchanged. CPU host/RNG differences are recorded separately; historical GPU timings are noncontemporary.

**CPU diagnostic quality outcome: FAIL.** Remaining screen tasks: 0. The configuration is implemented and its backend acts, but observed quality gates fail; the current evidence does not support replacing E22.

## Nonlinear learned generator, matched 2000 updates

| Package | Precision | Modes /25 | Mass TV | Toy gate | CPU train s | Updates/s | Ordinary moves | Iso moves |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| E22 | 0.6808 | 25 | 0.3192 | FAIL | 77.0514 | 25.9567 | 1981 | 0 |
| CB64-RA | 0.5160 | 23 | 0.4840 | FAIL | 35.6389 | 56.1184 | 3268 | 0 |

Toy useful-quality gate is precision ≥.9, all 25 modes supported at ≥1% of all draws, mass TV ≤.1. It is independent of native acceptance. Initial G/D/prior hashes exactly match the previous GPU study; current packages share CPU initialization and saved real data streams.

## MNIST convolutional GAN, matched 2000 updates

| Package | Raw embedding FD | Active embedding FD | Active precision | Active recall | Class TV | Confident classes /10 | CPU train s | Ordinary / iso moves |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E22 | 7.6781 | 0.6651 | 0.8345 | 0.8774 | 0.0631 | 10 | 339.8030 | 398 / 0 |
| CB64-RA | 16.7476 | 1.1627 | 0.9297 | 0.1519 | 0.0914 | 10 | 278.3647 | 5386 / 85 |

The reused real-only evaluator has 98.08% heldout MNIST accuracy. These are learned embedding metrics, not Inception FID. The previous 64-coordinate normalization had a heldout activation in a training-constant coordinate; raw 64 embeddings and the already declared 39 training-active standardized coordinates avoid that normalization artifact. No numerical MNIST quality gate was predeclared. Classifier confidence and embedding distance do not certify visual sample quality.

## Matched CPU training time

toy: common trainer time 35.64s. Latest eligible checkpoints:

- E22: update 750, 29.61s; precision 0.5262, modes 25, TV 0.4738; 6.03s unused due to checkpoint granularity.

- CB64-RA: update 2000, 35.64s; precision 0.5160, modes 23, TV 0.4840; 0.00s unused due to checkpoint granularity.

mnist: common trainer time 278.36s. Latest eligible checkpoints:

- E22: update 1500, 255.20s; raw/active FD 10.3840/0.7900, class TV 0.0685; 23.16s unused due to checkpoint granularity.

- CB64-RA: update 2000, 278.36s; raw/active FD 16.7476/1.1627, class TV 0.0914; 0.00s unused due to checkpoint granularity.

## Frozen scorer diagnostics on CPU

| Task | CPU scorer verdict | Completed updates | CPU seconds | Ordinary moves | Iso moves |
|---|---|---:|---:|---:|---:|
| mode_hold | PASS | 1200 | 19.6100 | 0 | 0 |
| img_intensity2 | PASS | 600 | 15.7600 | 4 | 0 |
| img_blobs4 | PASS | 600 | 15.0600 | 0 | 0 |
| img_stripes2 | PASS | 600 | 15.4900 | 2 | 0 |
| img_bars4 | FAIL | 600 | 15.3100 | 2 | 0 |
| vector_two_broad | PASS | 1200 | 26.1600 | 0 | 0 |
| vector_unequal_mass | FAIL | 1200 | 47.8500 | 12 | 0 |
| vector_unequal_width | PASS | 1200 | 31.5600 | 100 | 0 |
| vector_anisotropic | PASS | 1200 | 30.6400 | 2 | 0 |
| vector_overlap | PASS | 1200 | 29.5600 | 0 | 0 |
| vector_spiral | PASS | 1600 | 36.2600 | 0 | 0 |
| ring_shift | PASS | 4600 | 480.3100 | 303854 | 5723 |
| stationary | FAIL | 7500 | 708.4300 | 344916 | 1355 |
| grid100 | FAIL | 7000 | 792.9300 | 16017 | 1958 |
| rotated100 | FAIL | 7000 | 823.4100 | 28092 | 4217 |
| staggered100 | FAIL | 7000 | 823.6700 | 23584 | 2110 |

Portability counts: {'PASS': 10, 'FAIL': 3, 'ERROR': 0, 'PENDING': 0}. Native counts: {'PASS': 0, 'FAIL': 3, 'ERROR': 0, 'PENDING': 0}. These are CPU diagnostic verdicts; no historical 13/13 or native 3/3 is inherited by the candidate.

Portability failures: img_bars4 finishes with only 3/4 modes meeting the mass threshold; vector_unequal_mass passes the last individual metrics but has only one consecutive terminal passing check instead of five; stationary undergoes a late departure at step7340 and finishes with HQ .2949 and terminal passing suffix0. The full stationary7500 budget exposes that late collapse. Both clean and noisy versions fail these three tasks.

Each native task requires all 7000 updates, 34 observations, five final 20000-sample checks and a 100000-sample holdout. The unchanged official coverage and accuracy scorer outputs are in each screen task’s native-noisy/verdict.json; clean/EMA metrics are diagnostics.

### Native primary holdout and scorer verdicts

| Task | Noisy coverage / accuracy | Terminal accuracy | Holdout | Clean coverage / accuracy | Precision | Mass TV | Centre RMS /σ | Trace bias | Radial KS |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| grid100 | FAIL / FAIL | 00000 | False | PASS / FAIL | 0.9384 | 0.0393 | 0.1945 | 0.3763 | 0.1542 |
| rotated100 | FAIL / FAIL | 00000 | False | FAIL / FAIL | 0.9385 | 0.0231 | 0.1422 | 0.3729 | 0.1490 |
| staggered100 | FAIL / FAIL | 00000 | False | FAIL / FAIL | 0.9382 | 0.0289 | 0.2298 | 0.3345 | 0.1415 |

Required holdout precision ≥.97, mass TV ≤.06, centre RMS ≤.20σ, absolute trace bias ≤.10 and radial KS ≤.04, plus original mode/covariance coverage limits and sustained terminal checks. Clean diagnostics do not replace the noisy primary. EMA holdout scores are retained in the official verdicts; in these runs they equal the live holdout scores and also fail.

### CPU resources

| Run | Peak process RSS MiB | Timing scope |
|---|---:|---|
| toy/E22 | 718.5 | Training excludes evaluation/checkpoint I/O |
| toy/CB64-RA | 662.8 | Training excludes evaluation/checkpoint I/O |
| mnist/E22 | 1034.5 | Training excludes evaluation/checkpoint I/O |
| mnist/CB64-RA | 974.7 | Training excludes evaluation/checkpoint I/O |
| mode_hold | 609.7 | Whole screen including evaluations/scoring |
| img_intensity2 | 610.7 | Whole screen including evaluations/scoring |
| img_blobs4 | 611.6 | Whole screen including evaluations/scoring |
| img_stripes2 | 610.7 | Whole screen including evaluations/scoring |
| img_bars4 | 611.4 | Whole screen including evaluations/scoring |
| vector_two_broad | 608.5 | Whole screen including evaluations/scoring |
| vector_unequal_mass | 616.5 | Whole screen including evaluations/scoring |
| vector_unequal_width | 611.8 | Whole screen including evaluations/scoring |
| vector_anisotropic | 609.5 | Whole screen including evaluations/scoring |
| vector_overlap | 609.5 | Whole screen including evaluations/scoring |
| vector_spiral | 607.9 | Whole screen including evaluations/scoring |
| ring_shift | 726.6 | Whole screen including evaluations/scoring |
| stationary | 762.6 | Whole screen including evaluations/scoring |
| grid100 | 920.8 | Whole screen including evaluations/scoring |
| rotated100 | 923.2 | Whole screen including evaluations/scoring |
| staggered100 | 929.3 | Whole screen including evaluations/scoring |

RSS is each host process’s peak; concurrent aggregate and separate scorer children are not included. GPU peak fields are null because no GPU was accessible.

## Checkpoint continuation

E22/toy: PASS; semantic state equality True; next 10-update branch counters [{'dim_skips': 0, 'discoveries': 132, 'evals': 1, 'iso_acted': 0, 'iso_dup_skips': 0, 'iso_evals': 1, 'iso_flagged': 240, 'iso_moves': 0, 'iso_skipped': 1, 'matched': 0, 'moves': 0, 'normalised': 0, 'realised_births': 28, 'realised_deaths': 0, 'stale_resets': 0, 'stays': 104, 'waited_births': 28, 'waited_deaths': 0}, {'dim_skips': 0, 'discoveries': 132, 'evals': 1, 'iso_acted': 0, 'iso_dup_skips': 0, 'iso_evals': 1, 'iso_flagged': 240, 'iso_moves': 0, 'iso_skipped': 1, 'matched': 0, 'moves': 0, 'normalised': 0, 'realised_births': 28, 'realised_deaths': 0, 'stale_resets': 0, 'stays': 104, 'waited_births': 28, 'waited_deaths': 0}].
E22/mnist: PASS; semantic state equality True; next 10-update branch counters [{'dim_skips': 0, 'discoveries': 6, 'evals': 1, 'iso_acted': 0, 'iso_dup_skips': 0, 'iso_evals': 1, 'iso_flagged': 0, 'iso_moves': 0, 'iso_skipped': 0, 'matched': 0, 'moves': 0, 'normalised': 0, 'realised_births': 0, 'realised_deaths': 2, 'stale_resets': 0, 'stays': 4, 'waited_births': 0, 'waited_deaths': 2}, {'dim_skips': 0, 'discoveries': 6, 'evals': 1, 'iso_acted': 0, 'iso_dup_skips': 0, 'iso_evals': 1, 'iso_flagged': 0, 'iso_moves': 0, 'iso_skipped': 0, 'matched': 0, 'moves': 0, 'normalised': 0, 'realised_births': 0, 'realised_deaths': 2, 'stale_resets': 0, 'stays': 4, 'waited_births': 0, 'waited_deaths': 2}].
CB64-RA/toy: PASS; semantic state equality True; next 10-update branch counters [{'cell_discoveries': 4, 'cell_evals': 1, 'count_test_terms': 1600, 'dim_skips': 0, 'discoveries': 4, 'evals': 1, 'feature_distance_cells': 526080, 'feature_forward_rows': 3100, 'feature_rebuilds': 1, 'invalidated_cells': 3, 'iso_acted': 0, 'iso_dup_skips': 0, 'iso_evals': 1, 'iso_flagged': 474, 'iso_moves': 0, 'iso_skipped': 1, 'matched': 28, 'moves': 28, 'normalised': 0, 'ordinary_moves': 28, 'projection_products': 10514432, 'realised_births': 28, 'realised_deaths': 28, 'stale_resets': 69, 'stays': 0, 'waited_births': 0, 'waited_deaths': 0}, {'cell_discoveries': 4, 'cell_evals': 1, 'count_test_terms': 1600, 'dim_skips': 0, 'discoveries': 4, 'evals': 1, 'feature_distance_cells': 526080, 'feature_forward_rows': 3100, 'feature_rebuilds': 1, 'invalidated_cells': 3, 'iso_acted': 0, 'iso_dup_skips': 0, 'iso_evals': 1, 'iso_flagged': 474, 'iso_moves': 0, 'iso_skipped': 1, 'matched': 28, 'moves': 28, 'normalised': 0, 'ordinary_moves': 28, 'projection_products': 10514432, 'realised_births': 28, 'realised_deaths': 28, 'stale_resets': 69, 'stays': 0, 'waited_births': 0, 'waited_deaths': 0}].
CB64-RA/mnist: PASS; semantic state equality True; next 10-update branch counters [{'cell_discoveries': 6, 'cell_evals': 1, 'count_test_terms': 1600, 'dim_skips': 0, 'discoveries': 6, 'evals': 1, 'feature_distance_cells': 527552, 'feature_forward_rows': 3123, 'feature_rebuilds': 1, 'invalidated_cells': 5, 'iso_acted': 0, 'iso_dup_skips': 0, 'iso_evals': 1, 'iso_flagged': 0, 'iso_moves': 0, 'iso_skipped': 0, 'matched': 51, 'moves': 51, 'normalised': 0, 'ordinary_moves': 51, 'projection_products': 5268992, 'realised_births': 51, 'realised_deaths': 51, 'stale_resets': 86, 'stays': 0, 'waited_births': 0, 'waited_deaths': 0}, {'cell_discoveries': 6, 'cell_evals': 1, 'count_test_terms': 1600, 'dim_skips': 0, 'discoveries': 6, 'evals': 1, 'feature_distance_cells': 527552, 'feature_forward_rows': 3123, 'feature_rebuilds': 1, 'invalidated_cells': 5, 'iso_acted': 0, 'iso_dup_skips': 0, 'iso_evals': 1, 'iso_flagged': 0, 'iso_moves': 0, 'iso_skipped': 0, 'matched': 51, 'moves': 51, 'normalised': 0, 'ordinary_moves': 51, 'projection_products': 5268992, 'realised_births': 51, 'realised_deaths': 51, 'stale_resets': 86, 'stays': 0, 'waited_births': 0, 'waited_deaths': 0}].

Equality requires every semantic state section, including controller, lr_settle, birth_death and row_evidence. Only observational birth_death.last.eval_seconds is excluded; raw hashes remain diagnostic. This checks deterministic saved-state replay; integration separately supplied a nonzero-move continuation test.

## Activity, integrity and limits

Candidate ordinary feature-cell backend acted: True. Reference high-dimensional row window caps n_eff at 99, below 3×128=384; the gate cannot activate. Isolation and ordinary reaction counters are distinct.

Frozen source integrity: `{"E22": {"config_sha256_matches": true, "all_package_files_match": true}, "CB64-RA": {"config_sha256_matches": true, "all_package_files_match": true}, "original_local_frozen_sources_match": false, "scheduling_amendment_sources_match": true, "local_sources_match_with_authorized_scheduling_amendment": true, "protocol_matches": true, "cpu_amendment_matches": true, "original_harness_sources_match": true, "native_official_scorer_sources_match": true}`.

The coordinator-authorized scheduling amendment first ran independent one-thread portability/native queues beside two-thread saved-state replay. After replay completed, the two freed threads ran rotated100 and staggered100 under exclusive ownership: three native processes and one portability process, at most four active numerical threads. The original native queue collects grid100, waits for both peer receipts, then intentionally exits75 before rerunning their tasks. That orchestration handoff is separate from quality verdicts. Learned-model runs finished under the original matched scheduling. Only collector handoff branches changed; exact before/after and orchestration hashes are in scheduling-receipt.json. Candidate, quality runners, hosts and scorers are untouched.

One seed per task, no seed sweep or post-outcome controller tuning. Experimental K64/rank8/parent bounds add constants beyond original E22. The CPU adapter preserves scorer thresholds, but device RNG and native initial tensor hashes differ from the canonical GPU fixture. The original GPU benchmark remains unvalidated in this environment. S1b/14k, S4 variants and a complete S1–S6/admissibility claim are outside this acceptance run.

Receipts: candidate-freeze.json, CPU-AMENDMENT.md, cpu-screen.diff, cpu-adapter-receipt.json, training/*/*/config.json, screen/*/execution-receipt.json, archived-E22-native.json, archived-E19a-portability.json. Commands and unbuffered logs are launch.sh, launch.log and per-run run.log. Full metric curves, initial hashes, peak RSS, checkpoints and raw scorer verdicts are retained.
