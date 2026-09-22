# Smart descent v2

Research continuation of the first learned LR study. The controller reads only
gradient statistics: current RMS relative to its initial RMS, alignment with a
moving mean gradient, RMS relative to the moving mean, and the opposing role's
last normalized RMS. A bias term permits fixed offsets. There are twenty
learned coefficients, with separate G/D rows for LR and regularization actions.
Features have no task name, formulation, evaluation metric, loss or clock.

Every five updates, after backward and before the optimizer step, actions are
smoothed halfway toward their predictions. LR multipliers are bounded to
[0.25, 2], relative to either constant LR or the existing cosine schedule.
Regularization multipliers are bounded to [0.5, 2]. They affect the next loss
computation, not gradients already calculated. D controls the entire existing
gradient penalty; G controls ParticleRegularizer where the host uses it. Other
losses, betas, clipping, parameter-group ratios and update counts stay fixed.
Gradient observations use no Adam state, so the controller also accepts raw
SGD. The separate SGD study tunes its base rates without Adam normalization.

## Declared protocol

All previously inspected toys are **development data**, including the nine
behavioral hosts. No result from them is called fresh held-out generalization.
The ring development objective rewards full coverage, HQ >=90%, sustained
success, and early confirmation. It no longer selects on SW1 alone. A candidate
that cannot sustain ring success is screened before the remaining eight hosts.
Incomplete rows cannot pass. The final winner must pass all 29 original bounds
and sustain all nine toys, then minimize mean normalized confirmation step.

Three new transfer cases are specified in `study.TRANSFER` and written to the
protocol before search starts: six-mode ring with width64/depth2, 16-mode grid
with width128/depth3 and 32 particles, and an eight-mode ellipse with width64,
16 particles and R1+R2. They have 1,600 updates each. They are evaluated only
after policy freeze. Compare cosine, the learned policy, feedback-zeroed bias,
LR-only and regularization-only actions. Failed transfer is retained, and is
not a reason to refit on these cases while still calling them fresh tests.

Three generations of twelve policies use a separate seed0 perturbation RNG.
Every GAN initialization remains seed0; there are no seed sweeps. Identical
proposals explicitly reuse previous measurements. All proposals, errors, action
traces, source hashes, and 24-point live curves are retained. EMA stays separate.
Times include setup, evaluations and controller work; a single time observation
does not establish reproducible throughput. Production defaults are unchanged.

```bash
python -u -m benchmarks.smart_descent.study --output /tmp/smart-descent-v2 \
  > /tmp/smart-descent-v2.log 2>&1
tail -f /tmp/smart-descent-v2.log
python -u -m benchmarks.smart_descent.refine --source /tmp/smart-descent-v2 \
  --output /tmp/smart-descent-lr-only > /tmp/smart-descent-lr-only.log 2>&1
python -m benchmarks.smart_descent.freeze \
  --source /tmp/smart-descent-v2 /tmp/smart-descent-lr-only \
  --output /tmp/smart-descent-final
python -u -m benchmarks.smart_descent.evaluate --output /tmp/smart-descent-final \
  --reference /path/to/conceptmod > /tmp/smart-descent-evaluation.log 2>&1
```

The second search fixes regularization and ranks every candidate on all nine
development hosts. It uses two generations of twelve proposals, with initial LR
coefficient deviation .025. Its separate perturbation RNG is 731; GAN seed stays
0. The objective is `50 * failed bounds + 20 * non-sustained toys + mean normalized
confirmation step`. It reuses the original cosine control with a pinned parent
snapshot. The final transfer challenger is the fastest **nonzero** policy that
passes 29/29 bounds and sustains 9/9 toys across both stages. Selecting a nonzero
challenger does not displace cosine as the overall winner if cosine is stronger.

`standalone.py` subsequently fits LR feedback directly on constant base rates,
with no clock input or external schedule. Its three generations of eight
proposals use perturbation RNG 1731, fixed GAN seed0, the same full-suite
objective and LR-only actions. The earlier three transfer cases have already
been inspected at that point. Two new task/architecture cases are declared in
`standalone.TRANSFER` before its first run and remain reserved unless a candidate
passes and sustains all nine development hosts.

```bash
python -u -m benchmarks.smart_descent.standalone \
  --warm-start /tmp/smart-descent-final/frozen.json \
  --output /tmp/smart-descent-standalone > /tmp/smart-descent-standalone.log 2>&1
python -u -m benchmarks.smart_descent.damping_probe \
  --source /tmp/smart-descent-standalone --output /tmp/smart-descent-damping \
  > /tmp/smart-descent-damping.log 2>&1
```

The damping probe tests twelve stronger responses outside the local search's
initial coefficient scale: gradient growth or innovation, gain .3 or .7, on G,
D or both. It screens ring stability before the remaining eight toys and
retains the same two reserved transfer specifications. Screened rows cannot
pass the overall suite.

Freeze records bind the selected search's original transfer declaration and
fitting-source hashes. They separately record evaluation-source hashes and
reject changes to the numerical training/controller sources before evaluation.
Reporting-only changes can therefore remain explicit without relabeling the
source code that produced the fitting results.

The prototype retains a full moving gradient vector per optimizer. It uses CPU
scalar observations and serial scoped hooks; GPU overhead and online controller
checkpoint/resume are not implemented or certified.
