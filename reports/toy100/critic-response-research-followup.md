# Critic response research after the exact cold failure

Reviewed September24,2026 while the guarded cold acquisition gate runs.
These are proposed diagnostics, not tested replacement training rules.

[Unrolled GANs, ICLR2017](https://arxiv.org/html/1611.02163v4), sections2.3–2.4
and AppendixB.2, distinguishes evaluating G against an improved critic from
differentiating through its optimization. Its second gradient accounts for
the critic's response. In its zero-sum formulation that term disappears at
an exact critic optimum; the appendix reports that it becomes useful when
the critic lags. Merely increasing D's update count is a different method.

Our host is general-sum: sharp paired Rp plus cap for D, opposite paired Rp
through a spatial stencil for G. Therefore the zero-sum envelope cancellation
does not follow from D stationarity. The current bounded fit uses a partial
G gradient, and the earlier joint implicit-map experiments do not evaluate
this total derivative. A bounded next diagnostic could compare a single
virtual D step with and without its full chain term on identical tensors,
then check the derivative against finite differences. The virtual rule,
Adam metric, differentiability of cap/activation branches and extra query
cost must be explicit. Neither this paper nor the proposed diagnostic proves
that the current host needs this term or that it would improve acquisition.

[Projected GAN-CLC, May26,2026](https://www.nature.com/articles/s44387-026-00120-3)
uses added discriminator and content objectives and changing coefficients.
Its mechanism includes a linearly decaying discriminator-regularizer weight
and varying content weights; the authors do not claim a global-optimum
guarantee from this decay. It is not direct evidence for the present fixed
objective, constant-rate response rule. We do not prioritize this package of
new penalties/schedules over the isolated critic-tracking failure.

[Adaptive noise injection, accepted August27,2026](https://journals.aps.org/pre/accepted/10.1103/ch8n-wrv6)
introduces a learnable scale in a CDF-based discriminator output activation
and analyzes gradient scaling. The available abstract does not establish
time-uniform quality or the current capped relativistic game's stability.
The captured host has a coherent, nonzero generator field; no observed
vanishing-tail failure currently justifies a new activation/noise-scale test.

The next full host remains gated on saved-state evidence. Do not replace a
failed1200-update acquisition result with a selected final checkpoint, a
longer acquisition budget, or another seed. A finite hold and useful local
response cannot by themselves certify indefinite stability.
