# CI status observed during learned-controller research

The previous PR run at `279dfe5` failed two tests on Python 3.10–3.12.
[Python 3.12 job](https://github.com/255BITS/ParticleGAN/actions/runs/35754369037/job/106836293065)
reported 419 passing tests, eight skips and two failures.

The MoG config comparison expected the historical trainer's implicit cap target
to be absent. The API now exposes `reg_kappa=1.0` explicitly. The comparison now
resolves that same historical default before comparing every setting; no
experimental config or numerical threshold changed. The targeted MoG tests pass
locally (one optional SciPy test is skipped in the local environment).

The remaining `test_particle_native_2d` failure is inherited. Its existing
latent-joint EMA action MSE is approximately **0.249**, above the fixed **0.18**
threshold. The observation control remains collapsed at approximately **1.752**.
An isolated archive of base commit
`c62d4b248ea0ac275e2a8da75a16bdafc4fba49c` reproduces the same reported trajectory
and failure under the current Python 3.12.13 / Torch 2.13.0+cu126 CPU runtime.
The relevant experiment and its thresholds have not been changed in this PR.
This gate is separate from the extracted conceptmod behavioral suite and has
not been included as a controller-training target.

Reproduction uses the base package and experiment in an isolated directory:

```bash
mkdir -p /tmp/base-ci-repro
git archive c62d4b248ea0ac275e2a8da75a16bdafc4fba49c \
  particlegan experiments/toy_particle_native_2d.py | tar -x -C /tmp/base-ci-repro
cd /tmp/base-ci-repro
python experiments/toy_particle_native_2d.py --log /tmp/base-particle-native.log
```

The experiment exits with failure; this is not a green-CI claim. Raw local logs
were checked against the GitHub failure before documenting it.
