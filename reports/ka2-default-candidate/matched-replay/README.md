# Research versus public API: matched ring replay

The K3P and corrected KA2 public optimizer and penalty factories reproduce
their research ring experiments **exactly** when the setup is held fixed. The earlier public baseline
was a different experiment: it changed the prior, batch, initialization,
random streams and schedules. Its lower score was not evidence that this
component port failed to reproduce the research algorithm.

This comparison uses the archived training loop and evaluator. It replaces
the optimizer and penalty constructors with public `Recipe.make_*` factories.
It does **not** exercise `GANTrainer` or repeat the full 22-task suite.

## Completed measurements

| Formulation | Research hold | Public hold | Research recovery | Public recovery | Exact update comparison |
|---|---:|---:|---:|---:|---|
| K3P, public 0.8.0 | 120/120 | 120/120 | 28/81 | 28/81 | All 7,200 optimizer updates agree |
| KA2, corrected draft | 120/120 | 120/120 | 50/81 | 50/81 | All 7,200 optimizer updates agree |

Hold checks occur every ten steps from 1210 through 2400. Recovery checks
cover the fixed deadline window, 2800 through 3600. Recovery requires 81/81;
matching the research result does not turn its failing recovery into a pass.

For K3P, every recorded loss, raw and post-guard gradient, parameter tensor,
Adam moment/counter, applied learning rate and critic EMA parameter agrees.
All 30,813 random-operation records agree, including their order, shapes and
values. Every diagnostic score agrees with the original archived research
run. [Machine-readable K3P comparison](k3p-comparison.json).

KA2 also matches all 7,200 optimizer updates and 30,813 random-operation records.
Its trace additionally checks live/EMA buffers, the complete surprise-history
digest and length, gate, adaptive rate, and EMA update/skip/reseed counts at
every critic update. All diagnostics match the original KA2 archive.
[Machine-readable KA2 comparison](ka2-comparison.json).

**Selection:** KA2 retains the measured hold and improves recovery from 28/81
to 50/81. That is a reason to investigate it, not a completed winner gate.
Both recovery results fail; KA2's full 22-task qualification remains incomplete.

## A real KA2 rounding defect found during the audit

The public critic EMA used multiply/add to average all floating buffers.
KA2 changes the averaging rate; that arithmetic can move a fixed Fourier
frequency by one representable float even when both inputs are identical.
The research implementation averages parameters only, leaving those fixed
frequencies untouched.

The fix uses linear interpolation for floating buffers. Equal inputs stay
exactly equal, while changing BatchNorm statistics still average. Integer
buffers still copy. The parameter EMA arithmetic is unchanged. Regression
tests compare frequencies, outputs and input gradients against the frozen
parameter-only EMA through changing KA2 decay.

A 1200-step diagnostic with the pre-fix draft locates the first numerical
difference at critic update **831**, in the EMA Fourier buffer. Gradients and
parameters first differ at **832**; the scalar loss differs at **838**.
All 10,318 training-batch random draws match the research prefix. The shorter
budget changes isolated evaluation frequency, so this diagnostic does not
claim equal aggregate random audits or comparable recovery scores.
[Pre-fix evidence](before-fix-comparison.json). The corrected full 3600-step
pair above matches both training and evaluation exactly.

## What was held fixed

- Frozen CUDA host, width-96 networks with three hidden layers, critic Fourier
  width 3, 12 particles with four latent dimensions, batch 128.
- The same saved initial parameters, fixture SHA256
  `cb5ddaeb89cd32209466ae3bb0cd1c19eb4d16652128df9bfc825e618200dd08`.
- Native CUDA random streams and draw order, including a fresh real batch for
  the generator loss; identical evaluation RNG isolation.
- The 1200-step research scheduling horizon, network/prior floors .01/.05,
  input noise ending at 120, output noise reaching full strength at 240.
- 3600 training steps, the same target shift after step 2400, same update order
  and evaluator. FP32, deterministic algorithms, TF32 disabled, one CPU thread.
- CUDA-resident scalar Adam counters, as explicitly initialized by the frozen
  research driver. This preserves its arithmetic, including the critic's
  surprise measurement. Default `GANTrainer` counter placement is not tested.

The optimizer owns the bare critic; the public penalty evaluates its EMA
through a shallow copy of the host noise wrapper, sharing the original policy.
The generator factory receives `latent_table` explicitly because the archived
prior and side-loaded public package have different Python class identities.
No research mechanism hooks are installed in the public process.

K3P public source is commit `0ff9a7afe5dcb828239369446cfe71971bce687b`.
Each public result records hashes of the loaded package files. Each result
directory retains the host source diff, initial host signatures, complete
compressed update/random traces and compressed final result. The audit tools
accept logical `.json` paths and read the retained `.json.gz` automatically.
`worker-k3p.py` is the exact
K3P worker snapshot; `worker.py` adds detailed KA2 controller/buffer auditing.

The ring does not activate sparse-row damping or direct-particle response.
Their full internal histories are not covered by these trace comparisons.
The historical 22/22 ledger remains valid for its saved research configuration;
it is not a new public-package or KA2 qualification.

## Reproduce or audit

Run from the repository root with the frozen runtime path recorded in the
worker available. Extract the public K3P source once:

```bash
mkdir -p /tmp/particlegan-public-080
git archive 0ff9a7afe5dcb828239369446cfe71971bce687b particlegan | tar -x -C /tmp/particlegan-public-080
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python reports/ka2-default-candidate/matched-replay/worker.py --arm research --variant k3p --output /tmp/k3p-research > /tmp/k3p-research.log 2>&1
python reports/ka2-default-candidate/matched-replay/worker.py --arm public --variant k3p --output /tmp/k3p-public > /tmp/k3p-public.log 2>&1
```

Output directories must be new. Use `--variant ka2` for the exact retained KA2
source and current public implementation. Logs emit progress every 100 steps
and can be followed with `tail -f`.

Audit the retained K3P comparison without training or PyTorch:

```bash
cd reports/ka2-default-candidate/matched-replay
python compare.py results/k3p-research results/k3p-public --canonical results/k3p-canonical.json --output /tmp/k3p-verified.json
python compare.py results/ka2-research results/ka2-public --canonical results/ka2-canonical.json --output /tmp/ka2-verified.json
python verify_artifacts.py
```

CPU validation after the buffer fix: **1058 passed, 11 skipped, 18 subtests
passed**. Optional-dependency and GPU/opt-in skips are not benchmark passes.
The command used `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MPLBACKEND=Agg python -m pytest -q`. An initial GPU-exposed full-suite invocation
was interrupted after 50 failures involving unsupported deterministic CUDA
operations and CPU/CUDA device mismatches; its log is retained alongside the
passing CPU log. The explicit CUDA replays above are separate validation.
