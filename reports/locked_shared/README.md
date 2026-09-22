# Locked shared: measured behavior

**locked_shared meets 1/3 behavioral targets in this run.**

Every row trains. Verdicts use measured outputs only; no configuration gates.

| Toy | Variant | Measurement | Result | conceptmod parity |
| --- | --- | --- | --- | --- |
| two_pole | locked_shared | travel=0.514372; median slope=0.419652 | **PASS** | MATCH |
| two_pole | stranger | travel=0.000000; median slope=0.072553 | **FAIL** | MATCH |
| two_pole | thinned_cap | travel=0.694063; median slope=2.337441 | **FAIL** | MATCH |
| trajectory | locked_shared | identity MSE=0.233890 | **FAIL** | MATCH |
| trajectory | stranger | identity MSE=0.580628 | **FAIL** | MATCH |
| trajectory | nearest_stranger | identity MSE=0.335505 | **FAIL** | MATCH |
| ring | locked_shared | modes=3/8; HQ=49.49%; effective modes=2.722 | **INCONCLUSIVE** | MATCH |
| ring | cap_off | modes=0/8; HQ=0.00%; effective modes=0.000 | **FAIL** | MATCH |
| ring | vanilla | modes=4/8; HQ=67.24%; effective modes=3.508 | **INCONCLUSIVE** | MATCH |
| ring | fm_on | modes=7/8; HQ=83.06%; effective modes=6.538 | **INCONCLUSIVE** | MATCH |

Thresholds copied from the reference:

- Two-pole cloud: mean absolute travel ≥ 0.30 **and** median critic slope ≤ 1.0 (80 steps).
- Trajectory: same-seed identity MSE ≤ 0.02 (400 steps).
- Ring: ≥ 7/8 modes and ≥ 90% of samples within 3σ of a center (1,200 steps). ≤ 2 modes is FAIL; other results are INCONCLUSIVE.

These are different host experiments, not a complete config × toy sweep. Unrun combinations earn no result. All use the original seed 0 on CPU, one thread. VICReg 0.05 and host latent width 4 remain in trajectory/ring; cover 1.5 is a training term in trajectory and only a logged score in two-pole. Ring has no cover loss.

**Reference parity: 10/10 rows match; maximum absolute metric difference 0.**
Tolerance: relative 1e-6, absolute 1e-7. Reference runs use the same loaded ParticleGAN primitives, but the original conceptmod training code and constructors. This verifies the extraction and PR #36 builder wiring, not equivalence to every historical PyPI version.

PyPI particlegan-0.5.0-py3-none-any.whl: the four primitive source files (`GANLoss`, gradient penalty, particle prior, VICReg) are byte-for-byte identical to the files used for this run. Wheel and source SHA-256 hashes are recorded in `results.json`.

The checked-in conceptmod leaderboard's all-pass claim does **not** reproduce here. The original reference loops have the same failures. Keep the parity finding separate from acceptance: investigate the trajectory/ring results before claiming a behavioral sweep. Budgets, seed, thresholds and algorithms were not retuned to make this table pass.

Source: [conceptmod 5571213](https://github.com/HyperGAN/conceptmod/tree/5571213f5e8e129cfda45c785c3f30aad9c1d8c9/conceptmod/toys); [ParticleGAN PR #36](https://github.com/255BITS/ParticleGAN/pull/36).

Runtime: Python 3.12.13; PyTorch 2.13.0+cu126. Full precision metrics and source hashes are in `results.json`.

CPU toy results only; no GPU or downstream application transfer is claimed.
