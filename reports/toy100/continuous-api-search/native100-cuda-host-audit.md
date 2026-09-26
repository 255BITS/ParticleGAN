The canonical native100 CUDA host is the published `gpu-known-winner-control/simpler22_reference` route for grid100, rotated100 and staggered100. Its frozen config SHA is `4af9863a319378b362bfb925b161d9ae8b8b07c9ecf1a452bb645570e04b99b7`. This authority follows the exact config and execution receipts, not a selection by score.

The initial prepared RP5 adapter needed an initialization correction: its CPU constructors plus CUDA redraws did not reproduce the archived CUDA host. The owner applied that correction during this audit; the final delta review below finds no remaining source blocker in this scope. Runtime fixture assertions still need to pass before updates. The preparation map left backend authority unresolved; the archived worker resolves it by calling `torch.set_default_device('cuda:0')` before construction (`reports/toy100/gpu-known-winner-control/worker.py:22`). All three run declarations pin the exact current worker SHA `15b309dde720b63d90fac1e721f4e153ce30022ceb62db0e2d98da4c9c752a4d`. All three per-run source archive hashes and their 23 bound source files verified.

Canonical runtime is **Torch 2.13.0+cu126, CUDA 12.6, cuDNN 91002, NVIDIA RTX A6000**, revision `cf30153c4c131c8164ee7798e5022d810682e2cb`, deterministic FP32, TF32 off, threads/inter-op 1. RP5 ring receipts also record 2.13.0+cu126 / CUDA 12.6 / A6000. The older recovery comparator's 2.14/cu130 issue does not apply to these native receipts. No new execution tested reproducibility.

The authoritative source archive is `/ml2/hypergan/ParticleGAN-k3p-continuous-search/reports/toy100/gpu-known-winner-control/archives/simpler22_reference.tar.gz` (SHA `6820de970e8e5b78dace410d88d5a81a73aa38ed73e2912931c329adb790c781`). Within it, `benchmarks/toy100/train.py:345–408`, `lib/toy_models.py:58–77`, and `particlegan/particle_prior.py:43–81` establish this exact order:

1. Fork CPU and logical CUDA device 0 RNGs; seed both to 1234. Use CUDA default factory device for the following host model construction only.
2. Construct ordinary ParticlePrior(20000, 2): allocate FP32 on CUDA, then normal_(mean=0, std=1) on CUDA; this discarded draw must be retained.
3. Redraw prior.z uniform_(-5, 5) on CUDA.
4. Construct nn.Linear(2, 2) on CUDA: its default weight draw followed by bias draw must occur. Then copy identity weight and zero bias; overwrites consume no RNG.
5. Construct SimpleMLPDiscriminator(2, 128, 3, fourier=3) on CUDA. Deterministic Fourier buffer precedes Linear layers; then default weight and bias draws for each of Linear(14,128), Linear(128,128), Linear(128,128), Linear(128,1), in that order.
6. After ALL critic default constructors finish, traverse those four Linear layers in order: xavier_uniform_ each weight, zero each bias. Do not interleave construction and Xavier redraws.
7. Retain initial prior/G/D tensors and CPU/CUDA RNG receipts, require known prior/G hashes before any update, and compare complete new parameter hashes across survivor hosts.
8. Restore caller factory-device context; create each candidate through its unchanged public API, preserving its declared optimizer and serial-backward semantics. Do not inject optimizer counters or copy historical noise/LR schedules into the continuing learner.

The critic input width is 14 after Fourier features. Its four weight shapes are `[128,14]`, `[128,128]`, `[128,128]`, `[1,128]`, with biases of 128,128,128,1; total 35,073 parameters. All default layer weight/bias draws occur before the four Xavier weight redraws. The discarded prior normal draw and discarded G/D default draws still advance CUDA RNG and must not be skipped.

The three archived `native/<task>/summary.json` files retain the same initial raw tensor hashes:

| Tensor | SHA256 |
|---|---|
| Prior, FP32 `[20000,2]` | `05039118d87f7fd7e7425d8b43b71061fddffafe73f270cb8ca8f9c8dd708637` |
| G identity weight, FP32 `[2,2]` | `a666c95f0822c64e01580063e9bb27c629d4d0534e3163a9611738599f97df2a` |
| G zero bias, FP32 `[2]` | `af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc` |

These are hashes of contiguous raw tensor bytes, without the dtype/shape metadata used by `public_worker.digest`. Require all three before the first update. The prior range receipt is -4.999968528747559 to 4.999938011169434. The archives retain **no full initial checkpoint and no initial critic per-parameter hashes**. Source-defined reconstruction on the matching runtime is available; historical critic bit parity is not independently proved. Save all newly initialized tensors, individual critic hashes and CPU/CUDA RNG receipts, then compare both candidates' new fixtures. A mismatch with known archived hashes must stop the run for resolution, not trigger a score-based fixture choice.

Initial sampled cloud evidence is also retained: all tasks' `snapshots/step_000000.npz` live/EMA arrays are FP32 `[4096,2]`, raw payload SHA `07cc7aede04e8f5aa799119a11b4a0c10aa6daa29d18281150c565205aff4222`. The companion JSON records per-task initial target, full 20,000 target and 100,000 holdout target array hashes, read losslessly using only stdlib ZIP/NPY-header parsing. These are additional stream checks, not a full prior/critic fixture.

| Stream | CUDA seed | Rule |
|---|---:|---|
| Initialization | 1234 | Fork CPU/CUDA RNGs and preserve full constructor/redraw order |
| Training real data | 1234 | Dedicated generator; D batch then fresh G batch; indices then Gaussian noise |
| Historical global training noise reset | 1234 | Historical host source; candidate retains its own declared training streams |
| Observation target | 1635 | Fixed 20,000 draw per task |
| Observation output noise | 1636 | Global CUDA stream, forked/reset; paired live/EMA |
| Observation latent indices | 1637 | Separate stream, reset; sample with replacement |
| Holdout target | 2835 | Independent 100,000 draw |
| Holdout output noise | 2836 | Forked/reset global stream, paired live/EMA |
| Holdout latent indices | 2837 | Separate fixed stream, sample with replacement |

No 1901 noise offset and no observation-step seed offset apply to native100. Preserve all 34 observations: 0,1,10,25,50,100, then every250 through7000; snapshots retain first4096 of the actual20,000 draws. Retain full final samples, full final-five quality clouds at6000/6250/6500/6750/7000 and independent100,000 holdout. Existing numerical coverage/accuracy gates and live-primary policy remain the authority.

RP5 original preparation findings (original hashes are retained separately in JSON):

- `native_gate.py:23–27` keeps its ring recipe and all 15 public package files identical, with `total_steps=None`; the7000 budget belongs to evaluation.
- Original `native_gate.py:32,40–51` contained the CPU construction mismatch. Original `native_audit.py:28` checked its string label only. Both were corrected by the owner as described below.
- `native_gate.py:55–94` correctly separates observation latent and output-noise streams, pairs live/EMA, uses candidate-owned actual output-noise amplitude, restores module modes, and checks learner/caller-data state around evaluation. Its holdout adapter correctly uses2837/2836, with AccuracyEvidence generating target2835.
- `problems.py`, `metrics.py`, `gate.py`, `accuracy_gate.py`, `accuracy_evidence.py`, and `lib/toy_models.py` match the archived numerical host bytes. `accuracy.py` differs only in the unused `oracle_reference` calibration generator's explicit CPU choice; its grading functions are unchanged. Imported `_init_linear` and `evaluation_steps` are unchanged despite other learner-plumbing changes in current `train.py`.
- Bind trusted native config/worker/source authority in the revised receipt; self-hashed source plus an initialization label alone does not establish fixture equality.

The native host is affine G / critic width128 / seed1234. It is distinct from the recovery ring and the separate broader mode_hold task. Do not force the historical finite7000 recipe's LR/noise schedules, wrappers, or optimizer internals onto either continuing candidate. CUDA construction can be scoped to host model initialization; preserve each unchanged candidate's public optimizer/state/serial-backward contract. The same frozen initialization and observation contract applies to any next survivor, including the subsequent data-drift candidate.

The owner corrected preparation while this audit was finishing. The corrected `native_gate.py:43–62` scopes CUDA factory placement to host construction, keeps the exact normal/uniform/default/Xavier order, restores placement before trainer/controller construction, asserts known prior/G raw hashes before trainer construction and any update, records every G/D/prior parameter hash, and saves full `GANTrainer.state_dict()` plus caller-data RNG. The checkpoint includes global CPU/CUDA RNGs as well as package streams. `native_audit.py:28–31` validates the corrected constructor order and actual recorded hashes. `native-fixture.json` pins the canonical summary/worker and current host source; these receipt hashes were independently verified against their files. No package change accompanies this correction.

There is **no remaining source blocker in this reviewed scope**. Runtime prior/G hash assertions and cross-survivor complete-initial-fixture comparison remain pending; no run or historical critic bit parity is claimed. Corrected source hashes and the original preparation hashes are both retained in JSON.

No Torch or NumPy imports, training, GPU work, tests, installation, or active checkout edits were performed by this audit.
