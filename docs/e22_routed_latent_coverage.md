# Routed latent-trajectory coverage

This new public-API example asks whether increasing training trajectory support
from two to eight initial latents improves unseen-latent convergence while
keeping the particle architecture and native game unchanged. It is a generated
FP32 transfer diagnostic. Actual reduced/full Supra results are separate; this
example cannot establish a native optimizer bug, a new default, or SOTA.

The motivating qualified Supra audit found FIT240 MSE
`0.0008796560854929564` versus TEST240 `0.004852501390370583` at update12800.
Earlier FIT12 clean/noisy game directions help FIT but harm TEST. The independent
full-FIT **accuracy** gradient also opposes TEST (cosine `-0.12676040479199902`).
These facts support testing latent coverage and do not isolate a game-specific
failure. No measured accuracy direction is applied during this example.

Source status: unexecuted draft. The
[protocol](e22_routed_latent_coverage_v1.json) needs root source, backend and cost
review before freezing. No software, scientific or media PASS is claimed here.

## One changed factor

Both arms use the same frozen width32 conditional velocity carrier, rank4,
two sequential token-routing sites (`input`, `edit`), shared-Up gated V3,
bank32x4 and batch4. The branch is
`h=Down(x); m=h+tanh(Hh+b)+h*tanh(C code); y=frozen(x)+Up(m)`.
H/b and Up start neutral; C, Down, H/b, Up, all queries and particles remain
trainable. Initialization uses public `initialize_` with named CPU generators
before optimizer/EMA construction; the public particle cloud uses an explicit
`Normal(0,1)` bank exception. This carrier uses FP32 throughout, with autocast
disabled, and supplies no BF16 parity evidence.

Six source captions and their six target captions condition a single shared
frozen teacher. For each seed, one initial Gaussian16x16 tensor is reused across
all sources and both neutral/positive caption paths. Integrate Euler50/CFG3 and
record times0,.1,...,.9. Target velocity always uses the current context's z/t
and positive caption; training predicts a paired residual toward zero.

Baseline seeds7063/7064 produce240 ordered FIT records; seeds7063..7070 produce
960 expanded records. The original240 are an exact independently reconstructed
subset. Neutral and positive paths share t0, so distinct context counts are228
and912. Preserve those multiplicities. TEST39001/39002 has240 records/228 distinct
contexts, excluded from FIT and guard. FIT, guard and TEST pools are disjoint.

Both samplers have960 addressed slots. Decode
`slot=160*source+20*j+10*path+time_index`; baseline uses seed-index`floor(j/4)`,
expanded uses`j`. All slots and external draws are paired. Baseline's240 records
repeat four times; the treatment changes support, not the RNG seed. Both arms
consume4096 B4 contexts over1024 native updates. Coordinate units and the64-row
editing guard are fitted only on the original240 and remain identical.

## Native execution and fixed verdict

The public Recipe/prior/loss/optimizers/E22Policy/RoutedRows and routed hooks
are exercised through the existing caption example's update, observation and
checkpoint helpers. The recipe is `e22_routed`, N32/Z4/B4, initial output sigma.125,
auto birth/death and settled reopening. Learned noise, DV12, KA2, native rates
and structural feature guards remain active; output-error guard is disabled.
The branch/query LR is5e-4 in both arms. There is no MSE objective, output-gradient
update, output-based proposal score, metric selection, or metric stopping.

One penalty call runs per update: KA2 calls1..799 are pureA; call800 begins
the blend. The toy's224 later calls differ from Supra's mature12800 state.
Record actual phases799/800/801/1024. Genuine score observations occur at
896/928/960/992/1024; clean FAST serving has sigma0/perturbFalse.

Numerical PASS requires all declared checks:

- Baseline final TEST MSE >=1.10 common-original-FIT MSE, with positive FIT MSE.
- Expanded TEST RMSE improves by at least0.1% at all five endpoints; each source
  and each TEST seed has at most1e-6 RMSE harm. Final TEST-minus-original-FIT MSE
  gap shrinks by at least10% from a positive baseline gap.
- Saved expanded residuals improve native paired generator game by >1e-6 under
  both frozen terminal critics at all five endpoints, with four common panels.
- Bank, every dense bank row and router gradients are live on at least90% of
  updates2..1024; bank/router change; every site's C and Up norms are positive;
  learned owners, gradients and optimizer moments are finite. Terminal zero-code
  ablation worsens RMSE and both judge games by >1e-6 and every source's RMSE.
- Initialization/streams are matched, two-update versus1+1 fresh-owner public
  resume is exact, frozen owners stay unchanged, observations preserve native
  and caller state/RNG, and actual native phase witnesses match the declared law.

Oracle/destructive classifier controls run before learning. A complete failed
numerical test exits1 and retains its evidence. An incomplete/failed prerequisite
exits2. The score never controls training. A pass supports this support-size
fixture only; coverage expansion can help regressors generally and does not
prove defective native behavior.

## Reproduction after freezing

Use the project environment from the repository root. Root's external watchdog
must bind the frozen source/card/native cohort and enforce each whole budget.

```sh
CUDA_VISIBLE_DEVICES='' python -m examples.e22_routed_latent_coverage \
  --software-prerequisite --protocol docs/e22_routed_latent_coverage_v1.json \
  --out /tmp/NEW-coverage-software
CUDA_VISIBLE_DEVICES=0 python -m examples.e22_routed_latent_coverage \
  --run --protocol docs/e22_routed_latent_coverage_v1.json \
  --out /tmp/NEW-coverage-science
CUDA_VISIBLE_DEVICES='' python -m examples.render_e22_routed_latent_coverage \
  --observed /tmp/NEW-coverage-science/observed-training.pt \
  --producer-report /tmp/NEW-coverage-science/report.json \
  --producer-completion /tmp/NEW-coverage-science/completion.json \
  --protocol docs/e22_routed_latent_coverage_v1.json \
  --out /tmp/NEW-coverage-media
```

CPU prerequisite60s constructs exact data, validates scorer controls, executes
six native recovery updates and restores fresh initial states. It emits software
qualification and null scientific status; it supplies no score/rank. The GPU300s
envelope includes data, the same recovery,2048 quality updates, all observations,
both common judges, cleanup and final receipts. Budget feasibility is currently
unmeasured. Fixed work includes7320 small frozen CFG data/scale forwards,
1200 B4 clean FIT/TEST forwards,60 zero-code forwards,20 small camera forwards,
and5280 paired critic comparisons (10560 critic forwards,21120 percontext native
G-loss calls); native probes/candidate reruns add conditional work.

The separate CPU60 saved-array reader authenticates card/source/native/report
hashes, recomputes metrics/games/gates from actual saved arrays and native
retention witnesses, and exports the actual six-frame training GIF at
0/128/256/512/800/1024. It uses fixed axes and a zero residual target, with
genuine endpoint score points and no interpolation. The final camera is the
literal subset of saved TEST240. Native finiteness/recovery/gradient provenance
remains producer evidence; the reader does not construct a model or execute
ParticleGAN. Keep bulk traces/checkpoints/tensors/GIFs local, with compact
receipts and the single current goal readout after execution.
