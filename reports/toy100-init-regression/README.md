# toy100 regression under the default QR initialization (#194)

**Result.** On develop, `initialize_` silently replaced the benchmark's own
`xavier_uniform_` critic with QR weights at the *PyTorch-default* RMS. That makes the
critic 8x smaller at its output and gives it 4x smaller input gradients. The shipped
recipe (RpGAN-logistic + b_cap(1), Adam (0, .999), input noise .5→0 by step 700,
network-LR horizon cap 1600) then fails the 100-Gaussian gate on all three problems
(0/3). The fix keeps any weight that the host re-initialized away from the PyTorch
default scale. It restores **PASS 3/3**, with evals bit-identical to the old init.
Default-initialized networks are unchanged: they keep the same tensors and the same
research-hook parity.

Setup: `configs/toy100/constraints_simple_regularization.json` (the benchmark default),
seed 1234, 7000 updates, `cuda:1`. There is one run per (arm, problem) and no seed
variants. Every arm changes only D's construction-time weights, except `*_r2prior` and
`hook`, which also re-space the particle prior.

## Leaderboard

Gate = coverage gate and accuracy gate (holdout), as used by `benchmarks.toy100`.
"max grad" is the largest probed ‖∇ₓD‖ over the run, and it includes the step-1 value
of the untrained D (about 5.6 for the xavier init).

| arm | gate g/r/s | modes g/r/s | final HQ g/r/s | first 100 modes | max grad g/r/s | D init |
|---|---|---|---|---|---|---|
| **recipe_fixed** | **P/P/P** | 100/100/100 | 0.988/0.987/0.985 | 750/750/750 | 5.7/5.3/5.6 | **fix:** public recipe path. `initialize_` keeps host-initialized weights |
| old | P/–/P | 100/–/100 | 0.988/–/0.985 | 750/–/750 | 5.7/–/5.6 | xavier_uniform random, zero bias (legacy `None` pin) |
| fix_scale | F/P/P | 70/100/100 | 0.660/0.987/0.988 | –/750/750 | 10.0/3.9/3.6 | rejected fix: QR at the host's realized per-layer RMS |
| hook | P/F/P | 100/2/100 | 0.985/0.061/0.990 | 750/750/750 | 4.0/16.4/3.8 | develop's registry hook `--init batch_feature_zero` (QR at xavier RMS + R2 prior) |
| new_hid_old_ro | F/–/P | 97/–/100 | 0.931/–/0.990 | –/–/750 | 7.5/–/2.6 | new L0–L2 + old xavier readout |
| qr_xavier | F/–/P | 65/–/100 | 0.614/–/0.989 | –/–/750 | 5.5/–/3.8 | QR (same keys) at xavier RMS per layer |
| rand_default | F/–/F | 100/–/94 | 0.984/–/0.806 | 750/–/– | 2.1/–/4.9 | old xavier draw rescaled to torch-default RMS |
| old_hid_new_ro | F/–/F | 99/–/88 | 0.960/–/0.775 | –/–/– | 3.0/–/6.4 | old xavier L0–L2 + new QR readout |
| new_r2prior | F/–/F | 99/–/55 | 0.971/–/0.578 | –/–/– | 4.0/–/5.0 | new D + R2 re-spaced prior |
| recipe (develop) | F/F/F | 100/2/44 | 0.954/0.031/0.470 | 2500/750/– | 2.9/22.5/8.5 | public path on develop (= `initialize_(D, key=1)`) |

In `rand_default` grid, coverage passes and accuracy fails: mass TV is 0.061 against a
limit of 0.06. The `recipe_fixed` grid and staggered eval curves (modes and HQ at every
eval) are identical to `old`.

## Step 1: what the new init does to this D

The D is `SimpleMLPDiscriminator(fourier=3)`: layers 14→128→128→128→1 with LeakyReLU(.2).
The benchmark and `examples/100gaussians.py` both apply `xavier_uniform_` weights and
zero biases. Measured at construction on grid100 (`diag_init.py`):

| init | L0 rms / σmax | L1, L2 rms / σmax | readout rms / σmax | D(real) std | ‖∇ₓD‖ mean / max |
|---|---|---|---|---|---|
| old (xavier) | .118 / 1.67 | .088 / 2.0 | .129 / 1.46 | .247 | .88 / 2.8 |
| develop direct API | .154 / 1.75 | .051 / .577 | .051 / .577 | .031 | .22 / .62 |
| registry hook | .119 / 1.34 | .088 / 1.00 | .125 / 1.41 | .177 | 1.21 / 3.5 |

- `initialize_` takes rho from a fixed table of standard declarations
  (`nn.Linear`: U(±1/√fan_in)). It cannot see that the host re-drew the weights with
  xavier. It therefore shrinks the hidden layers 1.7x and the readout 2.4x, and it
  enlarges L0 1.3x. Biases are zero in all three inits, and zero biases are kept.
- The research hook captures the real `uniform_` bound of `xavier_uniform_`, so it builds
  QR at xavier RMS. The hook is what the 22/22 qualified. The direct API matches it only
  for hosts that keep PyTorch's default init, which is what the parity test covers.
- The "neutral batch-distance readout" has no effect here. It zeros only the batch
  columns of a `BatchDistanceDiscriminator` head, and this D is a plain MLP.

## Step 2: when the runs fail

- **The acquisition window is short.** Passing runs cover all 100 modes at the step-750
  eval, just after the input noise reaches 0 at step 700. The network-LR horizon cap then
  lowers D's LR from 4.3e-3 (step 1250: 2.5e-3) to 4.3e-5 by step about 1750. A run that
  has not acquired the modes by about step 1500 only creeps afterwards. For example,
  develop staggered reaches 3 modes at 1000, 14 at 1500 and 40–44 at 7000.
- **The failure signature is a critic spike during the anneal.** In every failing run the
  probed max ‖∇ₓD‖ jumps from about 1.4 to 5–18 within 50 updates, somewhere in steps
  275–500, and loss_d rises from .69 to 1–43 (the b_cap term). Examples: develop
  staggered at step 300 (9.6), fix_scale grid at 450 (10.0), and the simple-critic
  harness's hook grid at 500 (18.5). After the spike the gradient field stays sharp
  (3–5, against about 1.3 in passing runs) and acquisition stalls. Passing runs never
  spike.
- **Rotated departs after acquiring the modes.** Develop's recipe and the hook both
  reach 100 modes at step 750. Between steps 750 and 1250, still at full LR, they
  collapse to 1–2 modes with a D spike (12–16) and D(real) < D(fake). They never
  recover.
- So both failure modes happen during or just after the noise anneal and before the LR
  cap. None of them is a late departure after the schedule has settled.

## Step 3: isolating ablations

- **Readout scale is the systematic factor.** On staggered, every arm with a readout at
  xavier scale passes: old, qr_xavier, new_hid_old_ro, fix_scale and hook. Every arm
  with a readout at default scale fails: recipe, rand_default, old_hid_new_ro and
  new_r2prior. The random-versus-QR structure does not decide staggered.
- **Scale alone is not the whole story.** QR at the host's scale (qr_xavier, fix_scale,
  hook) still fails one problem in three, and which problem fails moves around. The hook
  passes grid here but collapsed grid to 44 modes in the simple-critic harness. That
  harness's probe drew from the training output-noise RNG stream, and nothing else was
  different. So under QR weights this recipe sits on a knife edge that a noise-stream
  perturbation flips. Old xavier passes under both streams. `rand_default` (the old draw,
  only rescaled) fails staggered: the recipe is also sensitive to D scale.
- **The prior re-spacing (R2) does not help.** new_r2prior still fails staggered with 55
  modes.

## Step 4: conclusion and fix

**Root cause.** Part of the new init is wrong for this host. The direct API overrides an
explicit host initialization (xavier) with the PyTorch-default scale. That contradicts
the documented "host's declared RMS" rule and differs from the hook that develop
qualified. The critic comes out 8x smaller in output and 4x smaller in input gradients,
with a readout 2.4x smaller. Acquisition slows or breaks before the recipe's LR horizon
closes. The recipe's tuning (b_cap 1, noise .5→0 by 700, horizon cap 1600) is also
fitted to the random xavier critic. With QR structure at the correct scale it still
fails one problem in three, so matching the scale is not enough.

**Fix (library, `particlegan/initialization.py`).** `initialize_` now also keeps a
parameter whose realized mean square is not a plausible draw of its standard declaration:
more than 8 sampling σ away, where σ = √(0.8/n) for a uniform draw and √(2/n) for a
normal one. Such weights come from xavier, kaiming or a manual rescale, and they are kept
the same way constant, identity and zero parameters already are.

- The rule is deterministic and consumes no RNG.
- It never fires on a default draw: 0 of 15,000 default Linear/Conv/Embedding tensors
  were flagged in a check.
- It leaves default hosts bit-identical, so the hook parity holds.
- The xavier toy100 critic and the xavier example networks now keep their init.

**Verification.**
- `recipe_fixed` passes 3/3 (HQ .988/.987/.985). Its grid and staggered evals are
  identical to `old`.
- `pytest -q tests`: 1084 passed and 6 skipped. This includes the hook-parity test,
  the registry init test and a new host-init test.
- develop's 22/22 is unaffected. It ran the registry hook (`batch_feature_init.py`),
  which this change does not touch. The direct API gives the same tensors as before for
  default-initialized networks, which covers all the standard witnesses.

**Residual risks.**
- The recipe is fragile to D init on this config. QR at xavier scale passes 2/3, and
  its fate flips under a noise-stream change.
- A user whose critic keeps the PyTorch default init gets the QR init. That combination
  is unqualified on toy100 with this recipe.
- Hosts with explicit custom init (for example `examples/five_modes.py`,
  `lib/sparse_models.py`, `lib/denoising_toy.py`) now keep their init, as they did
  before #194.

## Why develop reported 22/22

The three native tasks in `reports/toy100/batch-feature-init` used the same default
config, but ran through the frozen research runtime with the registry hook (`failure_worker.py`
installs the captured-declaration hook). The hook respects the xavier scale; the direct
`initialize_` API used by `get_recipe()` never ran toy100. In the current code the hook
gives 2/3 here (rotated collapses), and a different 2/3 in the simple-critic harness,
consistent with the knife-edge above; develop's frozen runtime landed on the passing side.

## Reproduce

```bash
reports/toy100-init-regression/run.sh old:grid100 recipe:staggered100 ...   # arm:problem, max 3 concurrent, cuda:1
tail -f reports/toy100-init-regression/logs/<arm>-<problem>.log           # one line per 250 updates + evals
python3 reports/toy100-init-regression/summarize.py                        # leaderboard from runs/*/*/result.json
python3 reports/toy100-init-regression/timeline.py runs/<arm> <problem> 250
PYTHONPATH=. python reports/toy100-init-regression/diag_init.py grid100    # step-1 table
```

`results.json` holds every run's result. The runs themselves (250 MB) are not committed.
Every arm except `recipe_fixed` turns off `_host_initialized`, so it reproduces develop's
`initialize_`.
