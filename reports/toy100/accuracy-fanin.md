# Fan-in parameterization probe on the 100-Gaussian grid

This scratch experiment asks whether the high-rate shared recipe can train the
100-mode MLP when its linear weights use the common fan-in representation
`W = raw / sqrt(fan_in)`. It changes neither the unlabelled target sampler nor
the public `Recipe` fields. The [probe source](accuracy_fanin_probe.py) builds
and Xavier-initializes the ordinary MLPs in the same RNG order as the
production runner, then replaces selected `nn.Linear` modules without drawing
randomness. Every run retains a copy of that probe source, its declared
config, an initial-parity receipt, complete checkpoint curve, 20,000-draw
terminal evidence, and an independent 100,000-draw holdout.

This representation *does* change effective optimization geometry. For Adam,
whose parameter step is approximately insensitive to multiplying the
gradient by a positive constant, the effective weight step becomes about
`LR / sqrt(fan_in)` even though the optimizer group still declares the same
`LR`. A hidden layer with fan-in 128 thus moves about 11.3 times less per
weight update than an ordinary `nn.Linear`; the first generator layer with
fan-in 4 moves about twice less. This is an explicit architecture-level
parameterization, not evidence that the public optimizer rate alone works for
an ordinary toy100 MLP. If applied only to toy100, it must remain visible in
the architecture receipt when claiming one shared training recipe.

The [focused parity test](../../tests/test_toy100_fanin_probe.py) passed before
the runs. The initial generator's largest effective-weight difference from
the same Xavier draw was 1.49e-8, output difference 2.98e-8, and input-gradient
difference 1.86e-8. Parameterizing the discriminator yielded output and
input-gradient differences of 5.96e-8 and 8.20e-8. Biases match exactly and
conversion consumes no RNG. Pure float32 `raw / sqrt(fan_in)` cannot map every
Xavier weight back bit-for-bit; these differences are floating-point roundoff,
not a different initialization distribution. The saved step-zero sample
clouds differ from the ordinary shared-v3 run by at most 1.19e-7.

The first two G-only (local evidence: `artifacts/toy100-accuracy/fanin-equalized/g-only`)
and G+D (local evidence: `artifacts/toy100-accuracy/fanin-equalized/g-and-d`) runs held
seed 1234, batch 2048, Fourier 2, and 7,000 updates fixed. Their common core
used LR .00425, D multiplier 1, prior multiplier 2, Adam β=(0,.99), `b_cap`
coefficient 6 at κ=1.25, prior regularization .05, LR anneal starting at 60%
to floor .05, output-noise peak .029 warming over 20%, and input noise .5
ending at 10%. Only the selected MLP parameterization differed.

| Run | Final live modes | Final live HQ | Final mass TV | Original terminal checks | Accuracy terminal checks | 100k live holdout TV | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G-only fan-in | 92 | 0.98610 | 0.16370 | 0/5 | 0/5 | 0.16438 | FAIL |
| G+D fan-in | 100 | 0.98630 | 0.09465 | 3/5 | 0/5 | 0.09209 | FAIL |
| G+D fan-in, transfer repair core | 98 | 0.98320 | 0.12645 | 0/5 | 0/5 | 0.13017 | FAIL |

G+D fan-in improves the final mode allocation from G-only, reaches all 100
modes, and passes the original coverage gate at steps 6500, 6750, and 7000.
It still fails the mandatory five-check streak: steps 6000 and 6250 fail.
Its final and holdout mass TV also exceed the frozen accuracy limit .06. G-only
never reaches all modes. The first two probes therefore do not establish a
shared 22-task solution.

One further G+D run (local evidence: `artifacts/toy100-accuracy/fanin-equalized/g-and-d-beta999-prior3-end02`)
used the transfer-side
`β2=.999 / prior×3 / input-end=.2` core (local evidence: `artifacts/toy100-accuracy/compatibility/repair-configs/v3_beta999_end02_outwarm02_a6_priorlr3.json`).
It retains all common fields from that manifest, uses the toy100 resource
batch 2048, and omits the manifest's unrelated staggered100 batch override.
It changes three core fields together relative to the first G+D run, so its
effect cannot be attributed to one field. The independently scored final cloud
has 98 modes, mass TV 0.12645, center RMS error 0.140σ, covariance trace bias
+0.0048, and radial KS 0.0104. Its 100,000-draw holdout confirms mass TV
0.13017. The original gate fails all five terminal checks and the accuracy
gate fails all five. The trial therefore closes this bounded architecture
screen without a promoted shared 22-task candidate.
