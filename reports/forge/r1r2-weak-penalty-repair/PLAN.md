# Fixed-task R1/R2 weaker-penalty comparison

The goal remains one global recipe passing all six original ordinary Tier 1
tasks and its own clock. Read compiled memory and the [closed score-law readout](../r1r2-score-law-repair/README.md).
Its complete 21 cases/GIFs, independent collection audit and exact 96-panel
density reconstruction are prerequisites. Preserve all historical evidence.

The [research review](research/weak-penalty.md) selects **only `reg_coeff: .1 → .01`**
on exact measured non-saturating AMSGrad m110. Weaker input-gradient pressure
may allow finer narrow-component critic discrimination during training.
Confidence is low. The coefficient is a historically declared tenfold contrast,
not inferred from Ring variance, local convergence theory or an optimum.
The [saved-gradient fixture](../r1r2-score-law-repair/research/regularizer-balance/README.md)
measures penalty/classification balance without identifying a causal repair.
Historical weaker-penalty pole failure and possible rotational instability are
explicit counterevidence. No smoothing arm or subsequent penalty grid is selected.

Classify this as `floor_constant`. Both positive real/fake zero-centered R1/R2
terms remain every update, with gamma/2 times their summed mean squared gradients.
Keep native non-saturating loss, global Adam AMSGrad, D/G+E/prior beta1 .5/.5/0,
beta2 .99, constant LR .0085, all multipliers 1, epsilon 1e-8, learned prior,
zero extra prior regularization/transport/noise/EMA and every other field fixed.
No API or scientific-source change is required.

| Role | Global recipe | New runs |
| --- | --- | ---: |
| baseline | Exact cached scheduled reference | 0 |
| non_saturating | Exact cached measured NS AMSGrad parent | 0 |
| weak_gamma | Exact parent with gamma .01 | 7 |

Preserve protocol seed 0/public deterministic initializer and each original
architecture, target/data law, actual initial tensors/ordered batches, initial
prior distribution/capacity, sampling, update cap, evaluation cadence and gate.
Checkpoint every consumed isolated constructor/data/training/evaluation stream.
Direct pole retains generator-owned coordinates; token has no sampled prior;
words retains its zero-width exception and reversed real joint encoder stream.
No replacement identity/zero fixture or source/initialization waiver.

Success requires all original gates, including five terminal Ring checks with
covariance≤.85, minimum eigen ratio≥.15, HQ≥.85, mass TV≤.15, 16 modes and 4,096
samples, plus every ordinary gate and this recipe's own independent clock.
Report acquisition and endpoint evidence separately. Any required or clock
failure rejects the whole recipe; no per-task winner mixing. Calibration remains
provisional and screening supplies no default promotion or Tier 2 credit.

Exactly **21 final cases: 14 exact cached controls and seven new runs**. Full
campaign reservation stays 7,560 seconds, each recipe allowance 2,520. The same
paid cap **8,000 seconds** includes all **210 original attempts / 6,405.0795450999285
seconds** and every concurrent full task reservation. Remaining paid headroom is
1,594.9204549000715 seconds. Preserve the original 900-second word allowance in
full; denial may serialize work, never shorten it or raise the cap.

Archive `/mnt/ml7tb/ParticleGAN-forge/r1r2-weak-penalty-repair-20261012` shares the
original queue symlink and all EIGHT archives' combined 8 GiB cap/5 GiB free floor.
Use one worker per GPU 0/1, one CPU thread and unchanged deterministic/TF32
settings. Source stays `25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`;
runtime stays `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
Commit registration and independent actual preflight before admission; execution
binds that full SHA. Complete and audit this finite contrast before designing
another. No retries, seeds, continuation, longer training or automatic matrix.

Commit compact metrics, receipts, sources and actual-training GIFs. Bulk logs,
metric streams, checkpoints and panels stay archived. The technique inventory
remains the sole current leaderboard; this comparison grants no qualification reuse.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-weak-penalty-repair/workflow.py prepare --manifest reports/forge/r1r2-weak-penalty-repair/manifest.json --registration reports/forge/r1r2-weak-penalty-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-weak-penalty-repair-20261012
# Commit registration/preflight before admission, then redirect output to archive/run.log.
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-weak-penalty-repair/workflow.py run --manifest reports/forge/r1r2-weak-penalty-repair/manifest.json --registration reports/forge/r1r2-weak-penalty-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-weak-penalty-repair-20261012 --source-commit COMMITTED_SHA --submit --drain --gpus 0,1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-weak-penalty-repair/workflow.py collect --manifest reports/forge/r1r2-weak-penalty-repair/manifest.json --registration reports/forge/r1r2-weak-penalty-repair/registration.json --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-weak-penalty-repair-20261012 --output reports/forge/r1r2-weak-penalty-repair/publication --media
tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-weak-penalty-repair-20261012/queue/events.jsonl
```
