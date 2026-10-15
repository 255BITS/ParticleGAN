# RA14 compatibility and preservation

The archived winner can be restored mechanically through the current public develop and private B native `Recipe`/`GANTrainer` interfaces without converting its state. The admitted grid100 probe restored every saved model, optimizer, policy, controller, backend, and RNG field exactly. Its original sampler produced bit-identical 64-sample clean and declared-noise outputs for both historical LIVE and EMA views, and sampling preserved the saved state. This is evidence for checkpoint and inference compatibility. Future training updates remain untested and include source behavior changes.

The prototype and evidence are isolated in `/ml2/hypergan/atlas-ra14-compatibility-20261010`. The historical package, checkpoints, configuration, qualification receipts, and current source repositories were not edited. All 104 frozen source hashes and all three checkpoint hashes remain valid. There were zero training updates, zero gradient steps, no publishing or merge, and no qualification transfer.

## Actual tests and launch lineage

| Test | Scope | Result |
|---|---|---|
| CPU software identity | grid100, rotated100, staggered100 × archived, public develop, private B | All 9 combinations PASS: exact G/D/prior tensors and fixed table rows, full old recipe preservation, normalized native-loader recipe agreement, unchanged saved input. Full CUDA-to-CPU restore correctly rejected. |
| Native GPU strict restore | grid100 × public develop 5737 and private B da9c346 | Both SUCCESS; all 23 non-recipe state fields exact, state keys equal, no recipe additions. |
| Original sampler identity | 64 rows per LIVE/EMA view, clean and declared-noise, both native targets versus archived implementation | All 8 target/view output array comparisons bit-identical; effective sigma and seed metadata exact. |
| Sampling state preservation | Both current targets | Every saved field, including recipe and RNG, unchanged after sampling. |
| Historical full inference recovery | Reused recovery-owner evidence; not repeated here | Original 3/3 noisy PASS, 0/3 clean PASS, 36/36 regenerated arrays exact. |

CPU checks exercise architecture and schema identity; they do not emulate CUDA RNG or establish scientific performance. Current-source native GPU restoration of rotated100/staggered100 was not run. Their CPU checks and the recovery owner's original three-layout checkpoint/inference evidence are retained separately.

The successful attempt is [RESULT.json](/ml2/hypergan/atlas-ra14-compatibility-20261010/admission/2831ae792e13580b/RESULT.json), key `2831ae792e13580b4d7fe548b519b6eb152c7b4ccc488df9027d098293a2a066`, with a 120-second worker limit and 5-second settlement allowance. Unmodified shared `PolicyCoordinator` admission assigned physical GPU 1 as logical `cuda:0`, one CPU thread, 4096MiB host reservation. The supervisor charged 7.456516 seconds; child work was 2.667332 seconds; the task's conservative success cost includes 5 seconds settlement,12.456516 seconds total.

Two preceding launch-only attempts failed before a scientific child started. The first source snapshot omitted `experiments.forge`; the second included Forge but omitted its top-level `particlegan.execution` import dependency. Their immutable snapshots, registrations, errors, and logs remain preserved. The unchanged coordinator recovered each as interrupted and charged its complete 125-second allowance because no durable measured terminal existed. Supervisor import closure was then preflighted successfully before the third attempt. Total retained ledger charge is 257.456516 seconds, and conservative task charge is 262.456516 seconds under the explicitly expanded 375-second lineage cap. Earlier external lineage charges remain unchanged and separate. The successful launch does not erase either failure.

## Source and state ownership

Public develop was verified against the remote branch and pinned at `5737ade47dca89b04d338ada20667781d1f2b5df`. The initially inspected local `origin/develop` at `2859975707eb3ea4d31958ad1b03e9e69e148a53` was stale; its earlier CPU result and selective snapshot are retained separately. Private B is the frozen `da9c3462ec54d082e25ecd9595d7b093598edd77` snapshot with digest `9b8f883840af01b005b87968a6c0bfa376da1d6960b7e814ca5eee6554e5de0c`. [Source bindings](/ml2/hypergan/atlas-ra14-compatibility-20261010/evidence/source-bindings.json) and [source inventory](/ml2/hypergan/atlas-ra14-compatibility-20261010/evidence/source-inventory.json) record the bytes and changed functions. Target packages are imported under distinct aliases so admission scaffolding and different implementations do not silently replace one another.

Restore begins with the original `{'trainer': ...}` schema4 envelope and its full saved recipe. The generic RA14 JSON declares N 12/batch 128/z 4; the historical native runner overrides those to N 20000/batch 2048/z 2. Copying that JSON into a current preset would therefore reconstruct a different experiment. The adapter constructs the exact saved architecture and `requires_grad` topology before creating optimizers: two-dimensional affine G; discriminator hidden 128, three hidden layers, Fourier 3; raw `ParticlePrior`. It supplies saved optimizer and penalty options and calls the target's actual strict loader. Saved logical device, dtype, stream topology, and RNG bytes are preserved; unsupported CPU restoration fails explicitly.

[State mapping](/ml2/hypergan/atlas-ra14-compatibility-20261010/evidence/state-mapping.json) enumerates the complete contract. It includes fast/served/EMA parameters, Adam/AMSGrad moments and counters, KA2/K3P penalty and EMA-critic anchor state, controller, feature calibration and population selection, birth/death FIFO and lineage, row evidence, stationarity, surprise, reopen guard, and named/global RNGs. The generator/table/noise versus critic roles and historical conditional feature-cells rate mapping are preserved: initial LRs `[[0.0010625,0.0085,0.0010625],[0.00425]]`, with generator/noise factor 0.25. Backend selection remains feature_cells and saved served source remains `averaged`. Saving and loading bare G/table weights would omit essential state.

Historical transient gradients, mixed per-module training modes, and absolute tensor version counters were not saved. No claim is made that these unavailable transients were recovered; they do not prevent the demonstrated completed-boundary inference restoration.

## Exact sampler contract

`ParticlePrior.sample` chooses rows uniformly with replacement and returns raw `z[idx]`. The saved recipe's `standardize=True` does not normalize this supplied particle table. Critic feature geometry has its own standardization. There is no output-unit normalization in the historical sampler.

Historical clean generation is `trainer._generate(model, latent,0.0,latent_stream,rows=rows)`. It retains the frozen DV12 feature-cells/controller latent perturbation before G; zero removes the additive output noise only. The noisy view is that same clean array plus `trainer.output_sigma() * torch.randn_like(clean)` after generation. Effective served sigma is 0.029 at these checkpoints, including the historical floor/clamp; using the learned `log_sigma.exp()` alone would change the law.

Original terminal evaluation uses a fresh CUDA latent generator seeded 1637 and a forked global output-noise seed 1636. The 100k holdout uses 2837 and 2836. LIVE and EMA each reset those streams separately. The adapter preserves model/prior modes and global RNG with `fork_rng`. It does not share the output-noise stream with latent selection. Current generic public `sample(output_noise=True)` draws the additive noise from its sampling stream and therefore does not reproduce this exact original RNG law.

At a completed historical boundary `trainer.G`/prior can already contain the served average. The saved fast parameters are restored separately. Original scorer names LIVE/EMA are preserved; historical LIVE must not be equated with private B's clean fast training iterate. During training `_serve_release` restores fast parameters before the D and G fake paths. Both paths retain DV12 perturbation and additive training output noise drawn from the training noise generator; `_serve_apply` subsequently installs the served view. Private B's cleanLIVE path uses a fast training model with `perturb=False`, raw prior generation, and no additive output noise.

The adapter requires an explicit `output_noise` Boolean. Its reusable native construction is:

```python
from pathlib import Path
from ra14_compat import RA14Archive, load_package, load_checkpoint, restore_native

root = Path('/ml2/hypergan/atlas-ra14-compatibility-20261010')
current = load_package(root / 'source-develop-current', 'current_ra14')
trainer = restore_native(current, load_checkpoint('grid100'), device='cuda:0')

# Explicit original sampler facade, with the natively restored current trainer.
view = RA14Archive('grid100')
view.trainer = trainer
samples = view.sample(64, output_noise=True, ema=False)
```

CUDA execution must use normal admission and its physical-to-logical mapping. This example follows the construction and sampler tested in the admitted probe; it is not an authorization to bypass admission or start training. The facade's `state_dict()` retains the original envelope. The pinned historical implementation remains its default.

## Mechanical port and algorithm changes

The historical KA2 implementation is byte-identical to current public develop's KA2. Private B adds symmetric BCap and selects it through its recipe; it does not turn the archived saved recipe into B. B's reg 0.5, phase-local critic progress sensor, orthogonal initialization/seed 0, discriminator Fourier 2, MoG kernel sigma 0.025, and clean fast LIVE output contract belong to a different experiment. B's output-noise configuration remains 0.029; its 0.025 is a latent MoG kernel width.

Even with an identical saved recipe, current continuation code has changed. Both current targets exclude the noise tester from output-floor settlement; the archive included all non-critic testers. Current feature-cell mean recovery allows partial occupied groups after an actual R1 surprise fire and filters inactive weights/empty EMA counts before transport ranking. Controller diagnostic materialization was made lazy while retaining the checkpoint representation; inference parity covers the tested restored state. Private B additionally has optional progress/control and penalty paths. Consequently exact endpoint restoration is insufficient to claim identical next updates or a fresh 7000-update trajectory.

The useful integration is an explicit archived-winner compatibility lane with its full checkpoint and serving contract preserved. The task10 [dual-contract harness](/ml2/hypergan/atlas100-dual-contract-task10-20261010/README.md) owns original thresholds, repeated terminal seeds, holdout semantics, and separate noisy/clean contracts. A current cleanLIVE task cannot inherit the historical noisy pass. Mechanical restore needs no optimizer or sampler-state conversion. Future algorithm work should change one field at a time under the appropriate unchanged contract, with critic coefficient 3→0.5 separate from KA2→symmetric BCap, and training-derived adaptive covariance separate from the original fixed 0.029 serving law.

Task12's admitted saved-array evidence reports that scaling only OUTPUT residual noise 0.029→0.025 fails all original terminal and holdout checks while preserving masses and 100 modes; roughly 90–92% of macro within-mode covariance comes from noise. This is useful causal evidence from another lane, not a result of this compatibility probe. Task13's training-data-only conditional covariance proposal changes the serving law and remains separately owned. Adding output noise cannot establish a cleanLIVE pass or repair allocation/tail failures by itself.

Remaining work is explicitly unlaunched: current-source GPU restore for the two other layouts, fresh historical/current update parity, current cleanLIVE/29-task qualification, and any long training. The present compact tests were sufficient to settle the native checkpoint/inference compatibility question without duplicating the recovery owner's full replay.

Machine-readable outcome: [HANDOFF.json](/ml2/hypergan/atlas-ra14-compatibility-20261010/HANDOFF.json). Primary prototype: [ra14_compat.py](/ml2/hypergan/atlas-ra14-compatibility-20261010/ra14_compat.py).
