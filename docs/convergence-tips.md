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
   state (in our case the latent particles too, not just G). Direct evidence
   that EMA is a *read-out*, not a training accelerator: at the exact step an
   EMA copy first passed the bar (hq 0.90), the live generator it was
   tracking measured hq 0.54 — the orbit's mean is good long before any point
   on the orbit is.
2. **LR anneal with a floor.** Cosine-anneal all players' LRs late in
   training. Two sharp empirical edges: annealing to exactly **zero destroys
   the run** (the system needs a small residual LR to keep tracking the
   equilibrium — 5% floor worked, 0% and 2% failed), and annealing from step
   0 **starves fast early progress** — keep full LR through the
   coverage/sharpening phase, then anneal.
3. **A sample-point critic penalty** (`recipe.make_critic_penalty`).
   Penalizing D's input gradient at real and fake samples damps the
   oscillation *in the dynamics* rather than averaging it away (the R3GAN line
   of work gives the local convergence argument). In our runs it was also the
   enabler for a high-resolution D (see §3).

These compose: the shipped recipe uses all three. A penalty alone (with a
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
  2. **Critic penalty with a sharp D from step 0** — the penalty caps how
     steep D actually gets *at the samples*, which keeps usable gradients in
     the empty space while the Fourier capacity handles fine structure at the
     modes. Better and schedule-free; this is what the recipe does. The
     penalty strength (`reg_coeff`) and gradient cap (`reg_kappa`) act as a
     sharpness-vs-coverage dial rather than a stability dial: too strong
     covers but plateaus soft, too weak sharpens but drops modes. The
     defaults (1.0, 1.0) are tuned on this benchmark; change them only with
     coverage and sharpness measurements in hand.

This capacity-vs-steepness distinction is the most interesting transferable
finding: Fourier features and a gradient penalty look like they pull in
opposite directions, but they control *different properties* of D and compose
into exactly the discriminator you want.

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

## 5. Overcomplete latents: a lever for weaker setups

Early in the campaign, before the other fixes, raising z_dim from 2 (= data
dimension) to 4 was the single biggest coverage lever: from "never reaches 100
modes even at 40k steps" to "reliably 100 modes by ~10k". The intuition: extra
latent dimensions give the generator room to route mass around itself instead
of tearing. 8 was worse than 4 — modestly overcomplete, not huge.

The shipped recipe keeps z_dim 2 (the data dimension) and still reaches
100/100 modes: the learnable particle prior and the critic penalty supply the
transport the extra dimensions used to. If a generator on a new problem seems
topologically stuck, a modestly overcomplete latent is still worth a try.

## 6. Relativistic pairing (RpGAN) is an LR amplifier

RpGAN's practical effect here wasn't direct quality — it was **tolerating
3–5× higher learning rates** without collapse, plus visibly better mode
balance. Consistent with the R3GAN paper's landscape argument. Without a
critic penalty it still oscillates (as theory predicts); the pair is what's
stable. Once the penalty is in, the speed comes from the regularized dynamics,
not from cranking LR.

## 7. Negative results (what not to waste time on)

All measured in this setting; several contradict common defaults:

- **Raw net LR is a nearly flat axis.** The baseline's problem was never
  "LR too low" — sweeping 1e-4→3e-3 changed little until it collapsed.
  Asymmetry (TTUR, D faster than G) mattered far more than scale. And once
  a critic penalty regularized the game, even TTUR shrank to a mild tweak.
- **Muon on the GAN nets was the worst thing tried** (collapse at any LR).
  Muon on the *particle table* worked (it's an embedding-table story, §4),
  but was ultimately beaten by plain Adam β1=0 + a penalized RpGAN
  objective.
  AdamW vs Adam is a literal no-op at weight_decay=0.
- **Capacity knobs were flat or harmful**: wider/deeper nets, more/fewer
  hidden layers — noise. Fewer particles (500–5k vs 20k) decisively hurt
  coverage; the particle cloud wants to be much larger than the mode count.
- **Bigger batches hurt** (512–1024 slower than 256 in most configs).
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
   structure (§3); add capacity *with* the critic penalty in place.
5. Once it converges: delayed LR anneal with a floor (§2.2) so training ends
   *on* the solution instead of orbiting past it — post-convergence blow-up
   is real and sudden.
6. Only then consider exotic optimizers, capacity, batch size — expect flat.

## Reference: the winning configuration (this repo)

`examples/100gaussians.py` defaults. A later 420-run study of the penalty's
*centering* (`FINDINGS.md`) replaced the campaign's zero-centered penalty with a one-sided
cap, `relu(‖∇ₓD‖ − 1)²` on reals and fakes at coeff 1.0, and doubled the base
LR: the cap damps the game just as well (any sample-point penalty does) but
leaves D usable slope below the cap, which buys sharper modes at an honest
core width — 100/100 modes and hq 0.986 at 7k steps, core σ ratio 0.866,
zero collapses over 5 seeds, bar (100 modes & hq ≥ 0.9) crossed by ~5.5k.
The example now trains the recipe's critic penalty (K3P, which starts as RMS
R1 plus a fake-side cap and hands over to one-sided caps with an EMA-critic
anchor). The shipped example trains 7k steps with a delayed cosine anneal for a
stable endpoint.

| Ingredient | Value | Why |
|---|---|---|
| Objective | RpGAN (relativistic pairing, logistic) | LR headroom, mode balance (§6) |
| Gradient penalty | recipe default critic penalty (K3P), coeff 1, κ 1, every step | damps oscillation; enables sharp D without flattening it (§2, §3; FINDINGS.md) |
| D input | Fourier features, K = 2 | resolve σ=0.03 structure from step 1 (§3) |
| z_dim | 2 (recipe default; data is 2-D) | prior + penalty supply transport (§5) |
| Optimizers | recipe optimizers (`make_optimizers`), betas (0, 0.999) | sparse particle table (§4) |
| LRs | G/D 4.25e-3, prior ×2, D ×1 (`get_recipe("gan")`) | mild TTUR only (§7) |
| EMA | 0.995 on G *and* prior, eval-only | sits on the equilibrium (§2) |
| LR schedule | full LR for 60% of run, cosine to 5% floor | stable endpoint (§2) |
| Particles | 20,000 for 100 modes | fewer decisively hurts (§7) |

Starting point for contrast (Adam β1=0.5, z_dim 2, plain D, no gradient
penalty, no EMA, no anneal): 86–92/100 modes and 30% hq after 12k steps, ~80%
hq ceiling at 60k, never converged.
