# Family defaults search: round one

The first frozen search completed **28 complete configurations across five
families**. None passed every required gate. The best observed KA2 configuration
passed **10/24** before failing unequal mode mass. These results identify next
research candidates; they authorize no public-default promotion.

All candidates retained the same **3 smoke / 19 quality / 2 endurance**
denominators. Failed prerequisites stopped subsequent training. Unreached tasks
remain UNKNOWN. The table is sorted by the number of passed required gates in
this one source/runtime cohort; equal scores use alphabetical display order.

| Family | Best observed configuration | Smoke | Quality | Endurance | First required failure | Family trials |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| KA2 | `093c6f2bd417` — LR .006375, prior rate1 | 3/3 | 7/19 | 0/2 | Unequal-mass mixture: rare component and component shape | 4 |
| BCap | `08689a73c551` — D rate2, prior rate4, penalty .5 | 3/3 | 5/19 | 0/2 | Mode hold:7/8 modes | 8 |
| GAN v3 release0.7 (MoG) | `1e266b5a2986` — LR .006375, prior rate1 | 3/3 | 5/19 | 0/2 | Mode hold:3/8 modes | 4 |
| K3P | `00b69f56de29` — LR .0085, prior rate2 | 3/3 | 5/19 | 0/2 | Mode hold:2/8 modes | 4 |
| R1/R2 | `302b6baa44f6` — LR .0085, D rate1, cosine start .2/floor .1 | 3/3 | 1/19 | 0/2 | Residual correspondence: MSE .148376 | 8 |

Selection is one entire configuration per family. It never combines successful
tasks from different configurations. The hash is a deterministic tie breaker,
not evidence that one failed configuration learns faster than another.

KA2's unequal-mass failure is more informative than its aggregate quality:
97.6% of samples are high quality and total mass TV is .0421, yet the least
represented component has only .598 of its target mass. The full-component
covariance eigenvalue ratio is .198, indicating insufficient local width in a
component. The distribution gate catches omissions that aggregate appearance
can hide. See the certified task metrics rather than interpreting HQ alone as
convergence.

The two strongest BCap trials retain7 of8 ring modes with HQ1/.99976. Four
half-rate critics fail the earlier movement gate; two other full-rate critics
fail trajectory identity. R1/R2's best config learns trajectory identity
(MSE1.31e-6) but loses residual correspondence. Another R1/R2 trial passes its
final movement bounds while having only a two-observation passing suffix,
short of the required five. These are measured numerical failure signatures and failed
bounds, not causal claims about an isolated optimizer knob.

The round consumed **1,587.149 paid task seconds** in156 certified attempts.
The worker used one task per GPU and one bounded CPU lane. Smoke/behavioral
tasks ran on CPU; reached vector/mode-hold tasks used the admitted CUDA lane.
No image, native100 or endurance task was reached in this round. The declared
worst-case reservation covered all configurations, while the execution stop
was10,800 paid seconds. Early failures saved that later work.

The executed source commit is
`05a2b3c021155fa72471b2293c3fd6d41d1e58a0`, digest
`9f89fa7cc7af552ef7e41405fab415abbd00acf3d3c6e4af8e4435fb2423142e`.
This runtime used Python3.12.13, Torch2.13.0+cu126 and RTX A6000 devices with
driver580.173.02. Historical results used a different runtime and cannot
supply missing passes or a causal before/after comparison. Other GPU services
were active, and full acquisition-time contracts are absent on some legacy
adapters, so there is no fastest-convergence ranking.

The provisional screen is useful for finding failures. Accepted calibration
and separately registered confirmation/robustness are still prerequisites for
shipping a family default. Exact parameter-capacity witnesses establish the
limited representability question; they perform no ordinary training and
provide no convergence credit.

Use the [single shared family leaderboard](../technique-inventory.md) for the
current selected rows. Each search report preserves every complete candidate,
gate, original receipt binding, unknown task and paid cost:

- [KA2](../configuration-search/ka2-family-defaults-round1-v1.json)
- [BCap](../configuration-search/bcap-family-defaults-round1-v1.json) and [failure readout](BCAP_ROUND1.md)
- [GAN v3 release0.7 (MoG)](../configuration-search/release07-gan-v3-mog-family-defaults-round1-v1.json)
- [K3P](../configuration-search/k3p-family-defaults-round1-v1.json)
- [R1/R2](../configuration-search/r1r2-modern-family-round1-v1.json) and [failure readout](R1R2_READOUT.md)

[The checked archive and restoration manifest](phase1-archive.json) bind the
original request/evidence/result bytes, source and queue logs. Compact proof
receipts and immutable numerical snapshots are committed for team review.
Bulk artifacts live on this machine at the manifest's path; sharing Git alone
does not transfer that archive. Independent regrading requires its original
receipts, not the compact display summaries.

The next search is a separately declared public-API Atlas/E22 cohort. It uses
the actual selected public sampler, full unchanged case budgets and an
additional first-acquisition/hold requirement. Its eight-case scores are kept
separate from this24-case MoG cohort. Its specification and representation
card must be frozen before ordinary execution.
