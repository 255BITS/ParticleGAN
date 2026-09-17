# Literature map: local adversarial training of recurrent memory

Research pass: 2026-09-16. No training code changed or experiments launched.
Starting point: round14 handoff, winner implementation audit, and completed
information probes. This is a focused literature review, not an exhaustive
novelty search. Proposed adaptations below are hypotheses, not published
guarantees for this architecture.

## Problem and constraints

Runtime is x_t = G(z, M_t, c_t), M_{t+1} = W_D(M_t, x_t), with a fixed particle
and advancing clock. The augmented state is (M, z, c). Training already includes
real-prefix BPTT and bounded generated branches with one generated write.
No MSE training, full generated training trajectories, analytic geometry cursor,
EMA, clipping, default B-cap changes, or seed-only experiments are proposed.

Separate three horizons: real observation context, generated forward exposure,
and gradient history. Detaching an arbitrarily long generated stream removes
long backpropagation but still generates a training trajectory. That is outside
the current constraint; it must not be presented as a free workaround.

The winner's speed probe R2 declines from .954 at handoff to .436 after32
generated writes and near0 after128, versus .946 after128 real writes. This is
loss of readily decodable information, not a proof of total information loss.
Round14 improved clean information without improving continuation.

## Closest research and boundaries

**GibbsNet (Lamb et al., NeurIPS2017).** Alternates inference and generation,
matching data-clamped and freely sampled joint latent/visible distributions.
Uses short sampled chains and reports one-step discriminator backpropagation
worked best. Its idealized stationarity result requires noise/support and exact
adversarial matching assumptions. This is not a theorem for our deterministic,
fixed-particle, clocked process, nor a guarantee of correct temporal ordering.
[Paper](https://arxiv.org/html/1712.04120).

**Generative Stochastic Networks (Bengio et al., ICML2014).** Learns a Markov
transition whose stationary distribution models data; provides a useful language
for local repair and repeated composition. Denoising/walk-back and stationary
sampling are not the same objective as preserving a particular temporal process.
[Paper](https://proceedings.mlr.press/v32/bengio14.pdf).

**Professor Forcing (Lamb et al., NeurIPS2016).** Adversarially aligns recurrent
hidden/output behavior under teacher forcing and autonomous sampling. Closest
to our train/runtime memory-distribution concern, but the original uses sampled
behavior sequences. A one-transition adaptation would be a new, weaker variant.
[Paper](https://papers.nips.cc/paper/6099-professor-forcing-a-new-algorithm-for-training-recurrent-networks.pdf).

**Predictive State Inference Machines (Sun et al., ICML2016), and Contrastive
Predictive Coding (van den Oord et al.,2018).** Predictive states give memory
meaning through observable futures; CPC learns future-relevant information by
contrastive prediction. PSIM learns filtering with observed inputs and uses
regression/data aggregation, so it is not an autonomous GAN solution. Our future
ranking already borrows related motivation; repeating it unchanged is unjustified.
[PSIM](https://proceedings.mlr.press/v48/sun16.pdf),
[CPC](https://arxiv.org/abs/1807.03748).

**Stable Recurrent Models (Miller & Hardt, ICLR2019), and transverse contraction
(Manchester & Slotine,2014).** Uniformly contracting recurrent models forget
initial conditions. Transverse contraction instead studies attraction toward a
periodic orbit while allowing motion along it. Useful design principle: preserve
process differences while correcting unwanted deviations. The continuous-time
limit-cycle results do not directly certify our discrete clock-driven network.
[Stable RNNs](https://arxiv.org/abs/1805.10369),
[Transverse contraction](https://arxiv.org/abs/1209.4433).

**RTRL/UORO.** UORO (Tallec & Ollivier, ICLR2018) approximates recurrent
sensitivities online without retaining an activation tape, trading exactness for
estimator noise. Irie et al. (ICLR2024) investigate tractable exact RTRL using
element-wise recurrence. These address credit assignment along visited states;
they do not provide exposure to unvisited generated states. Under our current
constraint they could affect real-prefix learning, but cannot supply long
autonomous credit without actually running the loop.
[UORO](https://arxiv.org/pdf/1702.05043),
[RTRL2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/74aec30590e07dbe2e29879f9df14fb2-Abstract-Conference.html).

**R2D2 (Kapturowski et al., ICLR2019) and TTUR (Heusel et al., NeurIPS2017).**
R2D2 identifies representation drift and stale recurrent states in replay.
TTUR studies distinct G/D learning timescales. Our current trainer re-encodes
after D updates, so replay staleness is not an identified bug. G can still face
a changing learned memory representation across optimizer steps. Neither paper
establishes that slowing our writer will improve autonomous retention.
[R2D2](https://openreview.net/forum?id=r1lyTjAqYX),
[TTUR](https://arxiv.org/abs/1706.08500).

**Teacher forcing and coverage.** DAgger (Ross et al.,2011) motivates training
on states induced by the learner, but uses learner execution and expert labels.
Huszar(2015) shows ordinary scheduled sampling can be statistically inconsistent;
this is not a proof that our adversarial feedback is inconsistent. Generalized
Teacher Forcing (Hess et al., ICML2023) blends inferred and predicted states to
control training gradients. It is a dynamical-system reconstruction method,
not a one-step adversarial replacement.
[DAgger](https://proceedings.mlr.press/v15/ross11a.html),
[Scheduled sampling critique](https://arxiv.org/abs/1511.05101),
[GTF](https://proceedings.mlr.press/v202/hess23a.html).

A recent related direction is GTF-DEER (Hess et al., May2026 preprint), which
parallelizes long-sequence dynamical-system reconstruction. It improves the
computational route to long-sequence training, rather than eliminating that
training requirement. Lower priority under our present constraints.
[Preprint](https://arxiv.org/abs/2605.12683).

## Research options, in recommended order

1. **Measure changing representations before altering optimizers.** On identical
   real histories and fixed particles, measure G output and predictive-readout
   changes immediately before/after a D update. Raw memory distances alone are
   coordinate-dependent. Compare the writer update contribution with G adaptation.
   If large, test slower writer learning or several G updates per writer update,
   separating writer from judging-head timescales. Existing old frozen/slow-writer
   configs mean this is a matched modern-recipe question, not an untried idea.

2. **Adversarial matching of the result of a write.** Explore comparing real-write
   and generated-write successor states conditioned on prehistory and future
   evidence, retaining the existing point/pair/mismatch GAN. One generated write
   suffices to construct a local training example. This differs from round14:
   it directly compares successor distributions, rather than only ranking future
   observations under each state. Specify writer-versus-critic gradient ownership
   explicitly: D's writer is itself part of the dynamics being aligned. An
   unconditioned memory critic can match population marginals while exchanging
   trajectories or collapsing information. Detached references and mismatched
   histories need controls; neither is a general collapse-proof guarantee.
   Under an ideal sufficient state and exact conditional matching, successor
   matching can be redundant; its practical value would be better finite-model
   training signals, not new ground-truth information.

3. **Preservation and repair in separate learned directions.** Measure the full
   one-step map F(M;z,c)=W(M,G(z,M,c)), whose Jacobian is W_M + W_x G_M.
   Candidate-only B-cap does not constrain this map. Compare perturbations that
   alter predictive identity with nuisance perturbations, using labels only for
   evaluation. Then consider a learned persistent/fast decomposition with an
   adversarial preservation incentive. Previous GRU/slow-fast size sweeps failed;
   the incentive and selective behavior must be the new ingredient. Avoid a
   blanket contraction penalty that encourages all memories to become identical.

4. **Audit temporal credit separately.** Real-prefix detach-length controls can
   hold forward states/context identical while changing gradient history. This
   tests encoding credit, not autonomous coverage. Only prioritize an RTRL
   implementation if such evidence shows credit assignment is limiting.

Fixed-z deserves a diagnostic throughout: real-prefix anchors pair histories
with independently selected particles, whereas autonomous M becomes correlated
with the same particle. Correct one-step averages over particles do not alone
establish correct repeated fixed-particle behavior. Joint state/particle probes
can expose that discrepancy without changing the particle formulation.

Evaluate all candidates with full warm/cold continuation passes, Q/lateQ,
motion retention, process decoding at0/1/8/32/128 writes, and G read/use tests.
Long paths remain evaluation-only. Better critic accuracy or memory decoding
alone does not qualify a winner. No sweep is selected or queued by this note.
