# Fast GAN Convergence: Transferable Notes

Notes from a ~300-run optimization campaign on the `examples/100gaussians.py`
benchmark (100-mode Gaussian grid, σ=0.03, learnable particle prior). The
campaign took the example from *never converging* (≈90/100 modes, 30% of
samples sharp after 12k steps, still drifting at 60k) to **full coverage with
>90% sharp samples in ~3.5k steps**, a >15× wall-clock improvement.

Everything here was measured in one toy domain with small MLPs, so treat these
as well-motivated starting points, not laws. They are ordered by how well we
expect them to transfer.

Convergence bar used throughout: all 100 modes covered AND ≥90% of samples
within 3σ of a mode center ("hq"). Steps-to-converge is the metric.

---

## 1. Diagnose coverage and sharpness separately

Mode coverage and sample sharpness are different failure modes with different
fixes, and they trade off against each other through the discriminator.
Conflating them into one FID-like number wastes tuning effort.

- **Coverage stalls** → the generator/prior can't *find* modes. Levers: latent
  dimensionality, prior mobility, relativistic loss, D smoothness.
- **Sharpness stalls** → the generator can't *tighten* onto modes it already
  found. Levers: D's spatial resolution, oscillation damping.

In our campaign, coverage was solved early and cheaply; the sharpness plateau
(stuck at 75–80% hq for tens of thousands of steps) was the real wall. Track
both metrics independently and you'll know which problem you actually have.

## 2. GAN oscillation is probably your plateau (and the fixes stack)

The classic result (Mescheder et al. 2018, "Dirac-GAN") is that vanilla GAN
dynamics don't converge locally — they *orbit* the equilibrium. In practice
this looks like a quality metric that rises, then plateaus while bouncing
±0.1–0.2 between adjacent evaluations, forever. Every loss family, optimizer,
and architecture we tried at constant LR plateaued this way; none of them was
the cause. Three fixes, in increasing order of principle:

1. **EMA weights for evaluation/sampling** (decay ≈ 0.995–0.999, scale with
   run length). The live weights orbit a good solution; the averaged copy sits
   on it. This alone took a never-converging config to converged. It is
   near-free — always do it. If your metric plateaus noisily, check the EMA
   copy before concluding the model can't do better. Average *all* learnable
   state (in our case the latent particles too, not just G).
2. **LR anneal with a floor.** Cosine-anneal all players' LRs late in
   training. Two sharp empirical edges: annealing to exactly **zero destroys
   the run** (the system needs a small residual LR to keep tracking the
   equilibrium — 5% floor worked, 0% and 2% failed), and annealing from step
   0 **starves fast early progress** — keep full LR through the
   coverage/sharpening phase, then anneal.
3. **R1 + R2 zero-centered gradient penalties** (the R3GAN recipe: RpGAN loss
   + R1 on reals + R2 on fakes). This damps the oscillation *in the dynamics*
   rather than averaging it away, and comes with a local convergence proof.
   In our runs it was also the enabler for a high-resolution D (see §3).

These compose: our final recipe uses all three. Note that R1/R2 alone (with a
standard low-capacity D) did **not** converge in this setting — theory
guarantees local convergence, not fast global convergence.

## 3. Discriminator resolution: spectral bias is a real bottleneck

MLPs (and CNNs, more mildly) learn low frequencies first. If the data has fine
structure — σ=0.03 modes on a ±4.5 grid here — a plain D physically cannot
represent the gradient that would sharpen G until very late in training.
Sharpness stalls and it looks like a loss/optimizer problem. It isn't.

**Fix: give D high-frequency capacity, but cap its steepness.**

- *Capacity:* Fourier features on D's input (`[x, sin(2^i πx), cos(2^i πx)]`,
  Tancik et al. 2020). With them, our D resolved individual modes from step 1
  — full coverage by ~1100 steps instead of ~10k. For image GANs the analog
  is discriminator resolution/blur schedules rather than coordinate features.
- *The trap:* a sharp D from step 0 **strands modes permanently**. A
  high-frequency D has near-zero gradient in the empty space between modes,
  so any mass that hasn't found a mode yet gets no signal ever. Even 2
  Fourier bands capped coverage at 43/100 modes. More capacity than needed is
  also bad (Fourier-3 was 1.5–2× slower than Fourier-2; match the frequency
  content to the data).
- *Two working resolutions of the trap:*
  1. **Coarse-to-fine schedule** — smooth D until coverage completes, then
     ramp the high-frequency features in (~3k steps). Works (8.7k steps,
     robust), but the ramp timing is a sensitive hand-tuned schedule.
  2. **R1/R2 penalty with a sharp D from step 0** — the penalty caps how
     steep D actually gets *at the samples*, which keeps usable gradients in
     the empty space while the Fourier capacity handles fine structure at the
     modes. Better (3.3–3.7k steps) and schedule-free. γ is a steep
     sharpness-vs-coverage dial, not a stability dial: sweep it log-scale
     (γ=0.01 → sharp but drops modes; γ=0.1 → covers but plateaus soft;
     γ=0.02 was the crossover here). Expect fine sensitivity.

This capacity-vs-steepness distinction is the most interesting transferable
finding: Fourier features and R1/R2 look like they pull in opposite
directions, but they control *different properties* of D and compose into
exactly the discriminator you want.

## 4. Sparsely-updated parameters need special optimizer treatment

Any large embedding-table-like parameter (our 20k-row particle matrix; token
embeddings; recommender tables) where each row gets a real gradient only every
N steps interacts badly with Adam momentum: β1 keeps moving rows that weren't
sampled, drifting the whole table. Symptoms: mode imbalance, coverage that
degrades over training.

- **β1 = 0** on such parameters was a reliable win (and harmless on the dense
  nets in this setting).
- Alternatively, *deliberately long* momentum (β1 = 0.99, or Muon with high
  momentum) also works — momentum acts as a gradient accumulator across the
  sparse updates — but it then interacts with LR schedules (long momentum won
  at constant LR, short momentum won once annealing was added). β1=0 is the
  simpler, more robust choice.
- Such parameters typically want a much higher LR than the dense nets (10×
  here), with a broad optimum (10–20×) — starved mobility shows up as missing
  modes, excess as instability.

## 5. Overcomplete latents help transport

Raising z_dim from 2 (= data dimension) to 4 was the single biggest coverage
lever in the whole campaign: from "never reaches 100 modes even at 40k steps"
to "reliably 100 modes by ~10k" with near-perfect mode balance, before any
other fix. The intuition: extra latent dimensions give the generator room to
route mass around itself instead of tearing. 8 was worse than 4 — modestly
overcomplete, not huge. If your generator seems topologically stuck, try this
before touching the loss.

## 6. Relativistic pairing (RpGAN) is an LR amplifier

RpGAN's practical effect here wasn't direct quality — it was **tolerating
3–5× higher learning rates** without collapse, plus visibly better mode
balance. Consistent with the R3GAN paper's landscape argument. Without R1/R2
it still oscillates (as theory predicts); the pair is what's stable. One
surprise: once R1/R2 was added, the extra LR headroom disappeared (higher LR
was strictly worse) — the speed now comes from the regularized dynamics, not
from cranking LR.

## 7. Negative results (what not to waste time on)

All measured in this setting; several contradict common defaults:

- **Raw net LR is a nearly flat axis.** The baseline's problem was never
  "LR too low" — sweeping 1e-4→3e-3 changed little until it collapsed.
  Asymmetry (TTUR, D at 1.5–3× G) mattered far more than scale. And once
  R1/R2 regularized the game, even TTUR shrank to a mild 1.5× tweak.
- **Muon on the GAN nets was the worst thing tried** (collapse at any LR).
  Muon on the *particle table* worked (it's an embedding-table story, §4),
  but was ultimately beaten by plain Adam β1=0 + the R3GAN objective.
  AdamW vs Adam is a literal no-op at weight_decay=0.
- **Capacity knobs were flat or harmful**: wider/deeper nets, more/fewer
  hidden layers — noise. Fewer particles (500–5k vs 20k) decisively hurt
  coverage; the particle cloud wants to be much larger than the mode count.
- **Bigger batches hurt** (512–1024 slower than 256 in most configs).
- **γ-annealing schedules for R1/R2** were not better than a fixed
  well-chosen γ, and cost robustness. Prefer the static value + log-scale
  sweep.
- **Instance-noise annealing and lambda sweeps on the variance regularizer**
  did nothing useful here (the VICReg-style weight had a broad optimum at its
  default, with 3–10× overweighting catastrophic).

## 8. Suggested tuning order for a new problem

1. Instrument coverage and sharpness as separate metrics (§1).
2. Add EMA evaluation immediately (§2.1) — it's free and de-noises every
   later comparison.
3. If coverage stalls: overcomplete latent (§5), check sparse-parameter
   momentum (§4), consider RpGAN (§6).
4. If sharpness stalls: check D's spatial resolution against the data's fine
   structure (§3); add capacity *with* R1/R2, sweeping γ log-scale.
5. Once it converges: delayed LR anneal with a floor (§2.2) so training ends
   *on* the solution instead of orbiting past it — post-convergence blow-up
   is real and sudden.
6. Only then consider exotic optimizers, capacity, batch size — expect flat.

## Reference: the winning configuration (this repo)

`examples/100gaussians.py` defaults. Converges (100/100 modes, ≥90% hq) in
~3.3–3.7k steps across seeds in the early-stopping harness; the shipped
example trains 5k steps with a delayed cosine anneal for a stable endpoint.

| Ingredient | Value | Why |
|---|---|---|
| Objective | RpGAN (relativistic pairing, logistic) | LR headroom, mode balance (§6) |
| R1 + R2 penalty | γ = 0.02, every step | damps oscillation; enables sharp D (§2, §3) |
| D input | Fourier features, K = 2 | resolve σ=0.03 structure from step 1 (§3) |
| z_dim | 4 (data is 2-D) | transport room (§5) |
| Optimizers | Adam, β1 = 0 everywhere | sparse particle table (§4) |
| LRs | G/prior-base 3e-4, prior ×10, D ×1.5 | mild TTUR only (§7) |
| EMA | 0.995 on G *and* prior, eval-only | sits on the equilibrium (§2) |
| LR schedule | full LR for 60% of run, cosine to 5% floor | stable endpoint (§2) |
| Particles | 20,000 for 100 modes | fewer decisively hurts (§7) |

Baseline for contrast (old defaults: hinge loss, Adam β1=0.5, z_dim 2, plain
D, no EMA, no anneal): 86–92/100 modes and 30% hq after 12k steps, ~80% hq
ceiling at 60k, never converged.
