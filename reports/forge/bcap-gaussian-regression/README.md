# BCAP/DualNorm 1D Gaussian regression

**Disabling autograd multithreading changes this exact Gaussian smoke from PASS
to FAIL. Rank truncation alone does not cause its failure.** Four matched CUDA
arms isolate the two changes on the selected BCAP/DualNorm configuration. With
multithreading enabled, both polar rules pass; with it disabled, both fail.
This conclusion applies to this recipe, task, fixed initializer and runtime.

| Polar update | Autograd multithreading | Original smoke gate | Confirmed acquisition updates | Endpoint CDF KS |
|---|---|---|---|---:|
| Full reduced SVD, historical | Enabled | PASS | 917 | 0.0429741 |
| Numerical rank truncation | Enabled | PASS | 459, 667, 792, 1000 | **0.0371191** |
| Full reduced SVD | Disabled | FAIL | None | 0.0692344 |
| Numerical rank truncation, current | Disabled | FAIL | None | 0.0646034 |

The endpoint is a descriptive metric. The smoke gate requires any scheduled
state to pass every distribution bound and an independent draw at the same
unchanged state. Its CDF limit is 0.05; it also requires finite outputs,
location/width bounds, all 1,000 updates and all 24 paired observations. There
is no terminal hold requirement in this Gaussian Tier 1 task. For example,
the historical arm passes at update 917, although its endpoint's independent
confirmation has KS 0.056603 and fails.

The archived V4 run was a genuine PASS, confirmed at update 917; V5 and V6
were FAIL. The recipe, target, architecture, prior, initializer, task execution
and evaluator match. The recipe hash is
`4233c1f2f7e9f64edd8401a42bb4afe15ccc1f357bd4bab2dc8239225fa7c60b`.
Historical runtime metadata does differ: V4 declares Python 3.14.7/NumPy 2.5.3,
while V6 declares Python 3.12.13/NumPy 2.5.2. Both declare Torch 2.14.0,
SciPy 1.17.1, RTX A6000, driver 610.57.04, deterministic execution and disabled
TF32. The within-runtime comparison removes this ambiguity: the current arm
reproduces V6, and the full-polar/enabled arm reproduces V4 **bit for bit** in
all 25 saved primary sample tensors and independent confirmation tensors,
their metrics, initial/final model and optimizer states, and every consumed
named RNG stream. The extra snapshot is the unscored initial state.

All four new arms use GPU 0, seed 0 and the public deterministic initializer.
The task remains N(2, 0.5²), batch 128, latent dimension 2, G `2→32→32→1`
with LeakyReLU, D `5→32→32→1` after Fourier2 features, and 256 learned MoG
particles of fixed sigma 0.1 without standardization. The single selected
global trainer recipe uses G/D/prior rates 0.012/0.018/0.03, zero momentum
and zero prior regularizer. Effective rates are constant. Initial models,
seen target sequence, prior, recipe and all initial/final named stream states
match across arms. Every arm completes exactly 1,000 G, D and prior updates,
with zero unintended RNG deviations and exercised finite/mechanism guards.

The two factors change the trajectory from the first update. Both aggregate
post-update gradient hashes and model hashes first differ at update 1 when
changing either factor. These hashes follow the entire G/D/prior update;
they do not identify a particular derivative or substep. This is not a
step-400 intervention or a lost training extension. Based on the earlier
Ring16 investigation, our inference is that autograd scheduling changes
floating point accumulation in the higher-order critic computation, and
subsequent normalized updates develop different trajectories. This Gaussian
comparison establishes the causal runtime factor without measuring which
individual floating point operation first differs.

Truncation is substantial even in this small network. Across the current
arm's 2,000 polar calls on 32×32 weight gradients, numerical rank ranges from
2 to 27 and averages 15.995. Each call excludes at least five directions.
The complete polar rule assigns unit singular values to those numerical null
directions. Truncation changes the optimizer rule, but it remains compatible
with passing this Gaussian under enabled scheduling. All matrices are at most
32 wide, so removing the old >1,024 Newton–Schulz branch cannot explain this
test. Bias and row-normalized prior updates retain their original rules.

Current serial execution reaches one complete primary pass at update 167:
KS 0.0479772, mean 1.968416 and standard deviation 0.502161. Its independent
confirmation has KS 0.0541849 and fails, so no confirmed acquisition occurs.
The final mean 1.967879 and standard deviation 0.497889 are close to the
target, but endpoint KS 0.0646034 still fails. Matching two moments is
insufficient for the full distribution gate; this run also has little margin
at its closest observation.

One reporting defect is retained explicitly. The truncated/enabled arm
completed and saved all training, samples, checkpoints and its original PASS
before a post-run assertion rejected its receipt. A lazy import of
`benchmarks` reset the ambient autograd flag to False, and the assertion
incorrectly included no-grad inference forwards. Every training update was
explicitly scoped to True inside the executed `_execute_step`; its checkpoint
records `serial_backward=False`. The fourth arm, using the identical update
scope with the corrected audit, directly records 1,000 enabled updates,
7,000 enabled graph-building forwards and 2,000 enabled `autograd.grad` calls.
Its 48 observation forwards are no-grad and inherit False; 1,000 critic-step
fake-generation forwards are also no-grad but run inside the enabled update.
No learned graph is constructed outside the scoped update. The third arm's
live counter totals and rank histogram were not saved and are reported as
unavailable. Its recovery only reads saved metadata and tensors; it adds no
training, sampling or qualification. No arm was rerun.

Keep the chosen serial execution policy and investigate a trainer recipe that
passes with it. This result does not justify an automatic rollback or a
per-task scheduling switch. Any replacement global recipe must receive the
same matched full-suite validation, including Ring16 and words. Before adding
another mechanism, inspect the saved Gaussian distribution and critic/prior
gradients to determine why this recipe has so little CDF margin. Preserve the
Tier 1 acquisition gate and test sustained hold in Tier 2.

The [frozen protocol](protocol.json) reserves four task allowances of 120
seconds, or 480 seconds total. The four complete arms consume 4,000 public
updates and 44.940 measured adapter-loop seconds. Complete wall time for the
third arm is unavailable after its assertion failure. These are separate
causal diagnostics, with no ordinary Forge qualification, Tier 2 unlock,
seed search, gate changes or global inventory edits. The [readout](readout.json)
and [verification](verification.json) retain exact results and source identities.
Actual-training GIFs use only the saved scored observations:
[current](media/current.gif), [old polar/serial](media/old_polar_serial.gif),
[truncation/enabled](media/truncated_parallel.gif), and
[historical factors](media/historical.gif).

From the repository root, reproduce an arm with the project Python environment
and an external wall-time cap. Use a fresh output directory; the four declared
arm names are `current`, `old_polar_serial`, `truncated_parallel`, `historical`.
The harness installs historical semantics only inside its diagnostic process;
production defaults remain untouched.

```sh
mkdir -p runs/bcap-gaussian-regression
timeout --signal=TERM 120s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  .venv/bin/python -u reports/forge/bcap-gaussian-regression/run.py \
  --arm current --output runs/bcap-gaussian-regression/current \
  > runs/bcap-gaussian-regression/current.log 2>&1
tail -f runs/bcap-gaussian-regression/current.log
```

The initial three arms execute commit `f8153c516`; the fourth uses the
audit-only correction `0b3ef2e96`. Exact executed runner hashes are recorded
per arm. [publish.py](publish.py) verifies existing outputs and renders the
GIFs without new inference or sampling. Raw logs, trajectories, checkpoints,
samples and reproduction inputs remain in the local ignored archive described
by [archive.json](archive.json):
`artifacts/bcap-gaussian-regression-v1.tar.gz`, SHA-256
`7f2535941949296cfcdbddf8930a8b791a852f1728c783db937a0d480960185a`.
