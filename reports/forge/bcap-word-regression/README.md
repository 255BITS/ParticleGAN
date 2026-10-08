# BCAP/DualNorm five-word regression diagnosis

**Both changes independently cause a word-hold regression for the selected
BCAP/DualNorm recipe under this frozen protocol.** The controlled CUDA factorial
reproduces the historical PASS with full polar updates and enabled autograd.
Changing only numerical-rank truncation fails the original hold; changing only
autograd scheduling to disabled also fails. Their combination reproduces the
current V6 FAIL exactly. This differs from the Gaussian diagnosis, where the
truncated/enabled combination passes.

All four word arms acquire at least one full numerical quality pass. The
regression is in retaining that solution over the original five terminal
checks. [PR332](https://github.com/255BITS/ParticleGAN/pull/332) and
[PR339](https://github.com/255BITS/ParticleGAN/pull/339) change the trajectory, not
the task geometry or numerical bounds. This report retains the original word
gate and gives no new ordinary qualification or default-adoption credit.

## Historical evidence

| Exact selected BCAP/DualNorm cohort | Verdict | Passing checks / 24 | Terminal passing suffix / required 5 | First passing update |
| --- | --- | --- | --- | --- |
| V4, before both defaults | PASS | 17 | 6 | 5,001 |
| V6, after both defaults | FAIL | 13 | 2 | 1,667 |

Both cohorts completed all 20,001 updates for G, E, D and the prior. Their final
quality is perfect: all five words, quality fraction 1, mass TV 0.0189453, exact
paired reconstruction 1, minimum reconstruction token probability 1, and NLL 0.
The V6 failure is the incomplete terminal hold: at update 18,335 it has four
generated words, mass TV 0.210547, incorrect reconstruction, and minimum token
probability 2.36e-27. The final two observations recover, which is insufficient
for the frozen five-check requirement. V4 has an inverse failure at 15,001 and
then six passing terminal observations.

The task is named `five_word_joint_acquisition`, but declares
`evaluation.kind=transfer_sustained` and `minimum_stable_checks=5`. It measures
acquisition plus terminal retention. This report preserves that question and
its original verdicts. A future Tier 1 acquisition / Tier 2 retention split needs
new explicitly declared gates; it cannot relabel these results.

## Matched conditions and causal factors

The resolved global recipe is identical in V4 and V6:
G/E rate 0.012, D rate 0.018, prior rate 0.03, DualNorm, momentum 0,
prior regularizer 0, and constant effective rates. Both execute the public
WordFixture with G 2→64→128→168, E 168→128→64→2, and joint critic
170→256→128→1. The batch size is 256. The explicit finite-vocabulary prior
exception is five learned, nonstandardized two-dimensional particle-cloud
points; it has no MoG width or mixture noise.

Recipe, prior, initialization hashes and named parameter seeds, RNG manifest,
task execution law, resources and numerical evaluation semantics match exactly.
The changed word evaluator binding names the shared helper's serial-autograd
revision. Its scoring code and numerical bounds are unchanged. The request
runtime bindings differ (V4 Python 3.14.7 / NumPy 2.5.3 versus V6 Python 3.12.13 /
NumPy 2.5.2); these historical bindings alone cannot isolate a cause.

The factorial runs all four combinations within the same current runtime:

| Matrix update | Host autograd scheduling | Hold verdict | Passing / 24 | Terminal suffix / required 5 | First full quality pass |
| --- | --- | --- | --- | --- | --- |
| Full `U Vh` | Enabled | **PASS; exact V4 control** | 17 | 6 | 5,001 |
| Truncated `U diag(s>tau) Vh` | Enabled | FAIL | 14 | 3 | 5,001 |
| Full `U Vh` | Disabled | FAIL | 14 | 0 | 3,334 |
| Truncated `U diag(s>tau) Vh` | Disabled | FAIL; exact V6 control | 13 | 2 | 1,667 |

Only full/disabled fails endpoint quality (four modes, mass TV 0.217188,
incorrect reconstruction). The other three have perfect final quality. An
endpoint-only comparison would therefore miss two of the hold failures.

Both rules use the same reduced exact SVD. The threshold is
`tau=max(matrix.shape)*finfo(computation_dtype).eps*s_max`. Every word matrix is
at most 256 wide, so this task already used exact SVD before PR332. Removal of
the old >1,024 Newton–Schulz path cannot explain its trajectory or cost change.
The rank audit records actual removed directions, gradients and updates at
updates 1, 2 and the 24 scoring checkpoints.

At update 1, the hidden critic matrices retain 10/170 and 10/128 directions.
The larger G/E matrices retain 5/64 or 5/128, matching the finite five-code
support. The current control removes 14,490 of 15,146 available directions
across 192 sampled audited matrix updates. This is a substantial optimizer
change: the old full polar rule gives unit magnitude even to numerical-null
directions, while truncation suppresses them. These counts cover the audited
updates, not all 20,001 updates. Frozen raw receipts call below-cutoff directions
`removed` even in the full-polar arms; those arms actually retain them. The final
reduction explicitly separates below-cutoff counts from actual removal.

The [frozen protocol](protocol.json) permits one 900-second attempt per arm,
20,001 updates, all 24 scheduled clean/live 1,024-sample observations, seed 0
and the public named deterministic initializer. All arms use physical GPU 1
(RTX A6000) sequentially. No seed trials, retries, longer training, annealing,
gate changes, additional scoring draws, or Tier 2 runs are included. A real
initial media observation would consume an extra evaluation batch, so GIFs use
the existing 24 scored states.

The isolated [reproduction harness](reproduce.py) calls Forge's public-component
adapter and grader. Only the diagnostic process replaces the polar function and
bypasses the outer `run_task` scheduling decorator to permit the enabled control.
It audits the mode inside G/E/D forwards (including critic calls used by the
higher-order penalty), optimizer steps and polar calls. It checkpoints every
named consumed stream and hashes the actual sequence of real-word batch indices.
Production defaults remain unchanged.

## Verification, interpretation and limits

The CUDA byte comparisons reproduce **all 192 saved observation tensors** and
**all 54 model, optimizer and consumed-stream checkpoint tensors** for both
controls: full/enabled versus original V4, and truncated/disabled versus V6.
The two ambient CPU/CUDA RNG restoration snapshots differ; the word task
consumes its independently checkpointed named streams. All four arms have the
same initial model hashes, full resolved recipe, task fingerprint and actual
real-word batch-index sequence hash. Their guards record 20,001 updates for
each role, finite state, exercised mechanisms and zero unintended RNG changes.
The actual scheduling mode matches the factor in every audited forward,
optimizer, polar call and update.

Truncation changes the first critic polar update even though the first critic
gradient hashes match; the subsequent G/E gradient changes in that same update.
This is direct evidence of the changed optimizer rule. Changing autograd mode
leaves the audited gradients, polar factors and post-update models equal at
updates 1 and 2, but they differ by update 834. The first unsaved difference is
therefore somewhere after update 2 and no later than 834. This study isolates
the scheduling setting's effect on the trajectory, but does not locate the
exact first floating-point accumulation that changes. No extra gradient probes
or continuation runs were added.

The original word fixture optimizes its joint adversarial objective; it has no
explicit reconstruction loss pulling E/G back to the paired inverse. Its five
prior codes continue learning while the encoder tracks them. Truncation
suppresses many network update directions while the prior retains its existing
row-normalized update rule and rate. That changes the balance of actual player
motion even though nominal rates are identical. A global rate rebalance under
the new defaults is a plausible next hypothesis, not an explanation established
by these four arms.

The historical request's Python/NumPy binding difference remains documented.
Exact reproduction of both original controls in the same current runtime
removes it as a necessary explanation for these word trajectories. The claim is
limited to this fixed seed-0 task and selected recipe; this is not a seed
sensitivity study or a universal ordering of optimizer variants.

All four arms completed for **2,963.14 paid execution seconds** within the
3,600-second reservation, with no retries. Each arm's log contains one emitted
cuSOLVER SVD convergence warning and Torch used its built-in more accurate SVD
fallback on the same CUDA tensors. Emitted warnings do not count every fallback
call. Raw logs and their hashes retain these warnings. Physical GPU 1 also hosts
graphics processes, so costs are reported rather than ranked as a speed study.

Use the [compact readout](readout.json) and [archive receipt](archive.json) for
metrics and provenance. Actual-training GIFs use only existing scored outputs:
[full/enabled](media/full-threaded.gif),
[truncated/enabled](media/truncated-threaded.gif),
[full/disabled](media/full-serial.gif), and
[truncated/disabled](media/truncated-serial.gif). Export adds zero observations,
samples or updates.

## Recommendations

Keep these diagnostic PASS results out of the current qualification board.
Neither changing only truncation nor changing only scheduling recovers the word
hold while preserving the other new default. Returning to both historical
settings restores this one task, but is not a demonstrated global repair and
would conflict with the user's chosen project scheduling policy.

For the intended Tier 1 "can it pass?" smoke question, all four arms already
reach the full quality goal. Declare a separate word acquisition task and put
retention in Tier 2 if that is the desired ladder; retain this original hold
task and its failures. If sustained learning is the next objective, freeze a
small global recipe comparison under the selected serial/truncated defaults,
keeping Gaussian, Ring16 and the other required tasks in the same comparison.
Do not anneal, change seeds or silently relax the existing bounds.

## Reproduce

Run from the repository root with the project environment. The harness requires
CUDA and a committed, unchanged execution source. Each output arm directory is
exclusive; it refuses to overwrite an earlier attempt.

```sh
mkdir -p runs/forge/bcap-word-regression
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  reports/forge/bcap-word-regression/reproduce.py --run truncated-serial \
  > runs/forge/bcap-word-regression/truncated-serial.log 2>&1
tail -f runs/forge/bcap-word-regression/truncated-serial.log
```

Repeat once for each of `full-serial`, `truncated-threaded` and `full-threaded`
under the frozen budget. Tail the corresponding `runs/forge/bcap-word-regression`
log. The default output is local and ignored. Tensor comparison, report reduction
and GIF export perform no new training or sampling.

Saved-byte analysis uses CUDA and the exact original V6 artifact directory:

```sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  reports/forge/bcap-word-regression/analyze.py --v6 /path/to/V6/1986a8fae42d494b94584b6e60c153e7
```

The training execution source is commit
`2e9808ee4a8324d88b172a766f3b9f3ab86bb0f3`; the exact source digest and per-file
hashes are retained in every arm's source manifest and the compact receipts.
Raw evidence can be hydrated from the verified local archive without rerunning
unchanged training. The [archive script](archive.py) preserves every recorded
byte plus the executed source; the [analysis script](analyze.py) reduces it.

Original V4 attempt: `1e7e6e136ff8496490b15318e7a5f04f`, source commit
`79fdf16d2ed880a9db1873245f150375e3be31b0`. Original V6 attempt:
`1986a8fae42d494b94584b6e60c153e7`, source commit
`45f056556503341bccf3ade0cd3365c5d0dadb91`. Bulk logs, scored tensors and
checkpoints stay in ignored local artifacts; compact receipts retain hashes.
