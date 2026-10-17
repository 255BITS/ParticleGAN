# Native policy capacity: CPU replay admission

All six current public Atlas/E22 native density cases have **SUPPORTED CPU
capacity** at their original numerical bounds. Each uses one 20,000-sample
capture with fixed seed 34002, the original full execution horizon and
**zero optimizer or fitting updates**. All six collapsed-cloud controls fail.
This is backend admission for exact CPU replay; it supplies no ordinary training
qualification, learned stability, acquisition timing or cross-device comparison.

| Family / case | Parameter geometry | Modes | HQ | Mass TV | Radial KS |
| --- | --- | ---: | ---: | ---: | ---: |
| Atlas / grid100 | axis_unique | 100 | .98785 | .0264 | .0090808 |
| Atlas / rotated100 | lattice | 100 | .98680 | .0264 | .0175924 |
| Atlas / staggered100 | axis_unique | 100 | .98785 | .0264 | .0090814 |
| E22 / grid100 | lattice | 100 | .98665 | .0264 | .0176901 |
| E22 / rotated100 | lattice | 100 | .98665 | .0264 | .0176927 |
| E22 / staggered100 | lattice | 100 | .98665 | .0264 | .0177548 |

The [CPU helper](policy_native_cpu_compatibility.py) reuses the declared
[analytic construction](policy_native_representation.py): identity affine G,
200 near-unique prior rows per mode and the original target-width learned
output-sigma parameter. A fresh public GANTrainer initializes policy/averages
from these parameters. One public first-real `begin_step`/`abort_step` prelude
retains actual backend selection and controller/reservoir observations; it
completes no training update. Atlas uses its selected bounded feature sampler;
E22 retains its reference DV12 sampler. Their CPU observations pass the unchanged
full original gate. State hashes verify complete fixture/training RNG purity;
the global CPU RNG remains unchanged during observation.

The entire capture and six controls complete in **110.447 seconds**, under the
120-second wall cap, on one CPU thread. No GPU, five-capture expansion or
100,000-sample confidence expansion is used. Existing GPU confidence receipts
remain separate diagnostics: CPU and GPU generators do not provide bitwise
interchangeable random streams.

Raw receipt, complete CPU states and actual arrays are retained outside Git at
`/ml2/hypergan/toy-policy-native-capacity-cpu-20261002`. Receipt SHA256 is
`25eac6e7decb685be0887d3ddd9dd72df19abc57bbe51de792e0f18946009dac`.
Every record binds current case/preset/sampling/source identities and state/array
hashes. Its single observation declares `samples_key='34002'`,
`evaluation_seed=34002`, `completed_steps=0`; each NPZ has key `34002`.
These records allow the strict CPU-only capacity loader to restore the public
checkpoint and reproduce its actual samples without interpreting CUDA RNG state.

```sh
timeout --signal=TERM --kill-after=10s 120s \
  python reports/forge/family-winner-round1/policy_native_cpu_compatibility.py \
  --output /tmp/policy-native-cpu-admission
```

Learning-rate/prior-rate-only grid reuse is restricted to unchanged zero-clock
serving semantics; different model/prior/controller/sampling or protected source
identities require compatible evidence. This witness cannot fill any ordinary
training PASS cell or qualify an entire family on its own.
