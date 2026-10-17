# Fixed G16 versus G64 public E22 diagnostic

The only arm difference is the generator caller batch: 16 versus 64. The critic
always receives 16 rows. Both arms use a fresh, identically initialized
**late-common profile**: global critic score gain zero, local gain 1/16, a learned
zero-initialized free-sign channel-energy head, antithetic generator loss,
generator applied-rate factor 1/4, beta1 zero, and native full routed E22 controls.
This does not reconstruct the mature Nova→Qwen critic or its training history.

From the repository root, in an environment with ParticleGAN's dependencies:

```sh
timeout 60s env PYTHONPATH=. python -u examples/routed_generator_batch.py \
  --output /tmp/routed-generator-batch
```

Use a new output directory. The frozen campaign is 500 updates per arm, one CPU
thread, at most 60 seconds for both arms together. The process exits nonzero for
failed quality gates, nonfinite values, ownership/lifecycle failures, source
changes, or an incomplete budget. A hard external timeout also exits nonzero.
Tail the output and the two arm JSONL files to see progress. No extension or
fixture/seed revision follows a failed campaign.

The generator sees only source/time. Sixteen fit contexts each have sixteen
hidden, symmetric nuisance labels: eight fixed spatial fields with RMS 0.15 and
their negatives. The conditional mean comes from a fixed seed4 teacher with
the same width4 spatial FiLM generator architecture and zero code columns.
That mean is representable in the generator family. The label's mode identity
never reaches G, E, or the router. Four separate guard contexts contain every
nuisance mode. Sixty-four untouched population contexts provide the metric.

For this exact finite population, squared error decomposes as

```text
population error = mean ||prediction − conditional mean||² + 0.15²
population excess = mean ||prediction − conditional mean||².
```

The means and excess are evaluation references only. Every optimizer backward
uses paired-error RpGAN, plus native KA2 for D. Native row controls use actual
nuisance-bearing fit/guard targets. No reconstruction/MSE gradient reaches an
optimizer. A Euclidean conditional-mean optimum does not establish that it is
the learned GAN game's optimum.

Pass requires both endpoint live population excesses at most 90% of their
initial excess, and G64 endpoint excess at most 90% of G16 endpoint excess.
All 500 public lifecycles, finite score-active gradients, all 128 dense row
gradients, independent live/KA2-EMA ownership, frozen BF16 host, exact common
caller streams, and organic row evaluation/probe counters must also pass.
Accepted moves and rejections are reported, with no fabricated minimum count.
Serving choices are reported separately; the primary gate is clean live error.

Both arms construct the same caller panel of 64 G row identities: the exact
D16 identities followed by48 draws from the separate G-data stream. G16 trains
on its first16 after D's update; the other48 are declared shadow draws. Both
draw the same64 G Gaussian fields and a separate common16 D Gaussian panel.
Internal DV12 perturbation consumption can
change with G batch size, so the diagnostic makes no claim of private-stream
perturbation identity. All controls remain enabled. Global critic features
remain present for routed DV12 although global-score parameters are inactive;
gradient health requires the local and channel-energy score paths.

The companion six CPU tests qualify source/fixture plumbing, public ownership,
one-step native sequencing, exact public resume and read-only evaluation. They
are not the 500-update scientific comparison:

```sh
python -m pytest -q tests/test_routed_generator_batch_public.py \
  tests/test_routed_generator_batch_identity.py
```

The publication command verifies every frozen source hash and records the
checkout commit. A documentation or publication commit is permitted when those
bytes remain identical; a changed scientific package file is rejected. The
original executed V2 driver and protocol are preserved under
`docs/routed_generator_batch_20261002_archive/`. The publication copy has not
received a new scientific execution; its scientific function bodies are
identical and its seven CPU contracts pass. The extra contract covers commit
portability and rejection of source drift.

The exact source hashes, named streams, gates, reference package commit and command
are in `examples/routed_generator_batch_protocol.json`. This diagnostic can
support or falsify a minibatch-conditioning hypothesis. It does not qualify a
new default or all E22 tasks. The separate [results report](routed_generator_batch_20261002_results.md)
records both the passed fixed500 toy and a matched real100 LPIPS improvement
that failed its minimum-gain gate and received no continuation.
