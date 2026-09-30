# Reverse-KL + chi-squared KDE drift does not replace LR decay

**No.** One frozen mixture field was tested. It holds a clean covered cloud and
can repopulate a missing mode there, then fails the dense warm fork. Cold
acquisition was not run.

## Idea

Cao & Wei, [arXiv:2603.10592](https://arxiv.org/abs/2603.10592), Table 1:
every KDE f-divergence velocity is the mean-shift gap
`(m_p - m_q)` times a density-ratio weight. Forward KL uses weight 1 (the
drifting field already suggested for a separate test). This candidate uses
the paper's mode-covering mixture instead,

`v = (p/q + q/p) (m_p - m_q)`,

the sum of the reverse-KL weight `p/q` and the chi-squared weight `q/p`.
The shared `1/h^2` factor is omitted, as in their `h^2`-scaled identity.
Bandwidth is the maximum real-to-nearest-particle distance, clamped to
`[0.07, 3]`. Each output step is capped at 0.1 and each latent step at 1
after a ridge solve in the current Adam metric. The prior moves only when
`KL(p||q) + chi^2(q||p)` strictly falls. No mode centers, clock, rest gate,
or critic-width controller.

## Cloud filter

`reports/toy100/chi_mixture_cloud_filter.py`, 0.13s, clean free particles.
Identical clouds have drift 0. Distinct 12-on-8 clouds stay at 8 modes / HQ 1
for 200 steps. A missing mode 6 is recovered at step 24 (centers) and step 22
(width-0.07 cloud), both finishing at 8 / HQ 1. During that recovery, clean
HQ hits 0: the bandwidth that sees the hole also pulls covered particles off
their modes, and they only return after the hole fills.

## Warm fork (same process, torch 2.14 CPU)

| Variant | Warm checks | Terminal modes / HQ |
| --- | ---: | --- |
| Scheduled identity | 200/200 | 8 / .99902 |
| Constant Adam | not all (min HQ 0) | 8 / .64526 |
| PR84 path, correction off | 196/200 | 8 / .99927 |
| Mixture pullback | 173/200 | 2 / .20068 |

The correction-off arm is not the archived torch 2.13 result (200/200). It
misses updates 1129–1132, minimum HQ .86621, then holds. The mixture arm is
worse on this same run: all 200 proposals are accepted, bandwidth median
.263 and maximum 2.47, and checks 1174–1200 fail, ending at 2 modes. A
single far real inflates the global bandwidth, the energy falls, and the
ring collapses. That matches the cloud filter's mid-recovery HQ 0, which the
every-step warm gate cannot ignore.

Cold trajectory, ring, and the 2400 hold were not started.

## Recommendation

Do not search quantiles or step caps for this global kernel. The bandwidth
that reaches a missing mode (~2.3) is the bandwidth that abandons covered
modes. A later rule would need a force that stays zero for reals already
inside a within-mode radius and acts only on the far remainder; this run did
not test that truncation.
