# Prior EMA relaxation: diagnostic-host 3/3

> **Canonical-host correction (September 28):** The 3/3 measurements below used
> a diagnostic host that drew a new real batch on each callback call. The
> original frozen host reuses the first callback batch. Under that original
> host, staggered100 is **FAIL**: coverage PASS and terminal accuracy 4/5,
> with step 7000 centre .20266σ. The diagnostic-host result below is retained
> as research evidence. The native100 task is still unresolved on the frozen
> host. See [canonical staggered receipt](canonical-staggered100/native-noisy-verdict.json).

This is a research candidate for PR #155. The diagnostic-host 7,000 update QR/noisy
native100 gates **PASS on grid100, rotated100, and staggered100** with one package
SHA-256, `64f82d9edba2a1422206b8474867cfdd35a793e42f24727c4e08fb79d0d532fc`.
Every task passes all five terminal live accuracy checks and its independent
100,000-sample holdout. The package source, overrides, run headers, verdicts,
rates, and compressed diagnostic traces are archived here. The
[machine-readable manifest](manifest.json) records the exact hashes and scores.

| Task | Verdict | Passing observations | First arrival | Final streak | Final live precision | Terminal centre RMS range (σ) | Holdout centre RMS (σ) |
|---|---|---:|---:|---:|---:|---:|---:|
| grid100 | PASS | 21/34 | 1000 | 15 | .98575 | .15325–.16057 | .12154 |
| rotated100 | PASS | 8/34 | 5250 | 8 | .97275 | .14149–.15670 | .12170 |
| staggered100 | PASS | 19/34 | 2250 | 10 | .97870 | .18323–.19232 | .15590 |

The centre limit is .20σ. Rotated precision has the narrowest margin: its
smallest terminal value is .9708 against the .9700 limit. All verdicts came
from the unchanged frozen coverage and accuracy scorers. The harness copy used for the 3/3 runs changed the real-batch callback
to draw a fresh batch on each invocation. It left the scorers unchanged, but
changed the training inputs. This difference was discovered during final
protocol review; these 3/3 results do not qualify as frozen-host passes. This package is not a project default.

## How the change works

The candidate keeps paired particle birth/death transport, two critic updates
per generator update with fresh real batches, the prior-driven critic rate
floor, the learned output noise, and the multiscale stationarity tester. Its
previous form passed grid and rotated but missed three staggered terminal
centre checks around the .20σ boundary. A paired common-random audit found
that evaluating the live generator with the existing EMA prior improved all
five terminal centre estimates; replacing only the EMA generator had much
smaller effect. This motivated a training-time prior adjustment.

When the prior tester makes a stationary decision and reduces its rate, the
existing full EMA handoff occurs. On subsequent updates while the last
decisive prior verdict remains stationary and there has been no data-law
reopen, the live prior moves toward its existing EMA by
`a = 1 - exp(-Δτ / b)`, where `Δτ` is the *applied* intrinsic learning-rate
increment and `b` is the tester's current intrinsic block length. The EMA is
updated in its existing order before this relaxation. There is no task name,
metric, seed selection, threshold lookup, wall-clock schedule, or new tuned
decay coefficient in the rule.

The averaging displacement is translated into the tester's positional
anchor. Its stored gradient blocks, correlation pairs, intrinsic time, and
optimizer moments remain intact. The birth/death output-space anchor is
recomputed after the prior moves. A drift verdict or data-law reopen suspends
relaxation. The package logs the optimizer's intrinsic block displacement and
the relaxation displacement separately. All three diagnostic sidecars have
zero serialization errors.

## Falsification and progression

- Disabling birth/death transport from initialization failed staggered100
  (1/34; final precision .9695 and centre RMS .2151σ). The useful mass
  correction from paired transport should be retained.
- Copying the EMA prior only at completed intrinsic blocks improved the
  staggered terminal checks from two passes to four, but the 6250 centre
  remained .20169σ. That candidate is a frozen FAIL.
- Continuous relaxation passed staggered100, then rotated100 and grid100.
  The final diagnostic repair changed only how invalid tester rows are
  represented in an optional log. The repaired package was rerun from
  initialization on all three gates; all passed with one package hash.

The earlier restart from a staggered checkpoint did not reproduce an
uninterrupted run exactly. All runs above were fresh from initialization, but
the diagnostic host used a different callback stream. A fresh canonical
staggered run failed its final centre check, so there is no frozen-host 3/3
claim for this package.

## Full 22-task matrix

The requested full matrix is **7 PASS, 2 FAIL, 13 ERROR**: native100 3/3
PASS; six vector gates 4 PASS and 2 FAIL (`vector_unequal_mass`,
`vector_overlap`); `mode_hold` and four image hosts ERROR because they pass a
real tensor while the two-critic implementation requires a fresh real-batch
callback; eight custom hosts ERROR at parity because particle birth/death is
not bound for those models. Errors occurred before a scored observation.
See [all22-results.jsonl](all22-results.jsonl) and the per-task results and error
logs in [all22/](all22/). These outcomes limit this package to the measured
native100 research task and do not support promotion to the project default.

## Reproduce a native gate

```bash
CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/tmp/pr38-default-env/bin/python /ml2/hypergan/lrfree-20260926/harness/screen.py \
  --package-root reports/toy100/lrfree-search/prior-ema-relaxation/package \
  --overrides reports/toy100/lrfree-search/prior-ema-relaxation/overrides.json \
  --task staggered100 --output /tmp/replay-pr155-staggered100 \
  --device cuda:0 \
  --candidate-options '{"eval_output_noise":true,"strict_streams":false}' \
  --cand cf1-d2-prior-ema-relaxation-final
```

Run the command from the repository root, substituting `grid100` or
`rotated100` for the other native tasks. `strict_streams:false` permits the
two extra fresh real batches; it does not alter frozen scoring. The archived
[package source](package/particlegan), [overrides](overrides.json), and
[evidence](evidence/) are sufficient to inspect the implementation and
all three verdicts.
