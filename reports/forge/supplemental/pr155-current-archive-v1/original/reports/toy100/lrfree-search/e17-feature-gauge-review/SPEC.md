# E17 critic-feature reparameterization audit

Written before executing `probe.py`. This is an independent CPU audit of the
unchanged `pkg-E17` source at
`/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17/particlegan`. It is not a
trainer candidate or a benchmark run. Astra reviewed the design first.

Use float64, `N=1024`, 2-D raw inputs, one NumPy `default_rng(20260929)` stream,
and one fixed birth/death private stream seed `771234` for every comparison.
Draw independent 1024-row standard-normal table `q` and real reservoir `R`.
The **null** case uses these unchanged. The **shifted** case adds `(2.5,0)`
to the first 256 table rows; `R` is unchanged. E17's own `maybe_apply` draws
its fake pool `F` from the table with the same private stream in each variant.
No task geometry, labels, gate values, or native logs enter the fixture.

The critic has a nonidentity affine hidden map `h=Ax+b`,
`A=[[1.2,.35],[-.25,.8]]`, `b=(.15,-.2)`, and scalar head
`score=(.7,-.4)·h+.1`. Compare this base with equivalent critics whose hidden
map is `h'=Sh` and head weight is `w'=S^{-T}w`. The transformations are
`diag(16,1)`, `diag(1,16)`, uniform `16I`, and the 90-degree rotation
`[[0,-1],[1,0]]`. Raw clouds, table geometry, controller initialization,
and private RNG state are identical between variants. Check critic logits
and the extracted pre-head features before comparing decisions.

Run E17's actual feature extractor, isolation test, and one full
`ParticleBirthDeath.maybe_apply` evaluation for each case and critic. A sidecar
reconstructs its kNN distances, pooled dimension, density-ratio `x`,
isolation scores and conformal p-values; its BH flags must equal E17's flags.
Record flag IDs, selected child/parent IDs, and final row positions. Repeat
every evaluation with E17's float32 shortlist disabled so exact float64 kNN
separates geometric sensitivity from shortlist precision.

The logits and expected feature transforms must agree within `1e-12`. The
uniform-scale and rotation controls should preserve decisions exactly and
continuous statistics within `1e-5`; a failed control is a numerical or
implementation finding. An anisotropic transform with changed exact-float64
flags or moves is a counterexample to parameterization invariance. If only
continuous statistics change, report statistical sensitivity without claiming
trajectory divergence. If neither case produces any isolation flags or moves,
the one-step decision comparison is inconclusive. This finite fixture cannot
prove general invariance. It also does not determine whether using a
particular learned-feature metric is admissible under the user's portability
rule.
