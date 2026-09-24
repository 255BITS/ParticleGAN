# Sample-anchor saved-state audit

The frozen sample-anchor saved-state filter passes all 44 selected PR84 failure checkpoints: each has eight modes and fixed-draw high-quality mass 1.0. This is a local rejection filter, not a native warm, cold, or long-hold result. Its additional group-anchor objective and neural fit change the training rule; the free-output fixed-center argument does not establish stability under sampled groups, shared generator parameters, or stochastic training.

The independent [saved44 audit](continuous-evidence/round6-sample-anchor-independent/saved44/audit.json) compares all 44 ordinary PR84 supports and update records against the original capture, including all 11 available full-state hashes. It verifies each arm's applied D/G/prior rates of 0.00425/0.00425/0.0085, final Adam moment counters, and identical original/candidate RNG and noise endpoints. The [selection audit](continuous-evidence/round6-sample-anchor-independent/saved44/selection-audit.json) recomputes all 44 fixed-noise grades and target/accepted whole-map anchor costs. Every correction selected a converged joint G/prior fit; all 44 native real batches yielded eight MST groups. The correction receipts report 132 phase-batch and 44 each owner/RNG checks. Raw branch outputs, declaration, and exact source bytes are in the [hashed archive](continuous-evidence/round6-sample-anchor-independent/saved44/manifest.json).

A separate [native cold one-update smoke](continuous-evidence/round6-sample-anchor-independent/cold1-unobserved/result.json) invokes the actual `run_legacy` host from its unmodified 1200-step recipe, then stops after one complete D/G/correction update and before EMA/checkpoint work. It does not use the saved-state continuation observer. Disabled sample-anchor and original PR84 full states match exactly (`ff17bc43…`); active sample-anchor keeps the D, both Adam states, EMA, RNG, and noise state identical to original. The active correction selects a converged eight-group joint fit, lowering its native-batch anchor cost from 17.3713 to 3.12943. All three arms use one D and one G/prior Adam step, with the same constant rates. The adapter's inherited PR84 `locals()` phase binding is still part of the mechanism; this smoke rules out an extra observer dependency, not all instrumentation sensitivity.

Reproduce the audits using `/tmp/pr38-default-env/bin/python` and one CPU thread:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -m reports.toy100.pr84_reallocation_saved44_independent_audit --saved-filter /ml2/hypergan/ParticleGAN-continuous-learning/artifacts/continuous-learning/round6/sample-anchor-saved44 --capture /ml2/hypergan/ParticleGAN-stationary-diagnosis/artifacts/continuous-learning/stationary-failure-replay/capture-v2 --output /tmp/sample-anchor-generic-audit.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -m reports.toy100.pr84_sample_anchor_saved44_audit --root /ml2/hypergan/ParticleGAN-continuous-learning --saved-filter /ml2/hypergan/ParticleGAN-continuous-learning/artifacts/continuous-learning/round6/sample-anchor-saved44 --generic-audit /tmp/sample-anchor-generic-audit.json --output /tmp/sample-anchor-selection-audit.json
```

The cold smoke is source-bound to the archived declaration and is deliberately a single update:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -m reports.toy100.pr84_sample_anchor_cold1_unobserved --root /ml2/hypergan/ParticleGAN-continuous-learning --saved-filter /ml2/hypergan/ParticleGAN-continuous-learning/artifacts/continuous-learning/round6/sample-anchor-saved44 --output /tmp/sample-anchor-native-cold1
```
