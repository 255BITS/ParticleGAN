# CB64-RA: independent cost and geometry acceptance

**Verdict: the frozen candidate does not pass the required fixed families.** Its bounded native reaction passes the nominal speed gate (20.48× at N8192; time slope 0.501). Nominal parent selection misses the original relative mass-TV limit, high-dimensional detection misses every planted unsupported row, and configured latent jitter damages folded geometry.

Executed the actual candidate package/config and unchanged E22 on **40 fixtures / 80 backend rows in 127.531 s**, CPU with two threads. The separate six latent-sampling probes below add no acceptance rows and change no original verdict. There were no source/config fixes, seed sweeps, gate overrides, forced reaction markers, or threshold tuning.

## Original leaderboard

Cost columns: exact centers / common jitter first / common jitter four / native full first / native full four. Geometry columns: exact / common four / native full four. Counts require the original detector and every applicable frozen gate.

| Family / critic | Backend | Cases | Pass counts by scope | Ordinary moves, four | Isolation moves, four |
| --- | --- | --- | --- | --- | --- |
| cost:frozen_toy_head | cb64_ra | 12 | 4/4/4/0/0 | 5594 | 1382 |
| cost:frozen_toy_head | e22 | 12 | 6/6/6/6/0 | 599 | 1389 |
| geometry:trained600 | cb64_ra | 14 | 1/0/1 | 103 | 1188 |
| geometry:trained600 | e22 | 14 | 0/0/0 | 562 | 964 |
| geometry:frozen_initialization | cb64_ra | 14 | 1/1/1 | 3466 | 1624 |
| geometry:frozen_initialization | e22 | 14 | 0/0/0 | 13 | 1321 |

CB64-RA achieves selected-case improvements; no complete cost or geometry family qualifies. E22 also fails all geometry cases. A failed reference does not qualify the candidate. The frozen-initialization critic is a separately reported representation diagnostic; it does not replace the trained600 result.

## Scope and metric meanings

| Scope | Actual action | Support evaluation |
| --- | --- | --- |
| Exact isolation | Actual detector and parent selector; ordinary children empty; exact clones | G(table centers), with no copy jitter |
| Common isolation jitter | Four rounds; actual flags/parents; E22 DV12 cap; σ=.0125 geometry or .025 cost | G(table centers) after these copies |
| Native full reaction | Four unforced FIFO turnovers; actual ordinary + isolation + _move; candidate σ=.025, norm cap .05 | G(table centers) after native copies |
| Six sampling probes | Actual perturb_latent on every exactly repaired center, once | G(jittered latents); output-noise support undefined |

Native fake features include the actual configured latent jitter and output σ=.029. Original native quality receipts evaluate clean G(table centers), which include the copy perturbations already made by _move. They do not estimate the law served by jittering every row. The separately preregistered probes resolve that particular gap. Output-noise support is undefined under this latent/G semantic oracle and is not counted as a pass.

Repair recall is repaired original unsupported rows / all original unsupported rows. Support-repair precision is written original unsupported rows that were repaired / all written rows; semantic versions additionally require the original intended mode. In full reaction that precision includes ordinary relocations of supported rows. Its low value alone does not establish harmful ordinary behavior; the population support, TV, rare allocation, retention, and move types below provide separate evidence. Exact isolation-only support repair recall/precision are 100% in the eight detected cost fixtures, and 0 in the four highdim no-ops. Realized and expected intended-mode parent validity are conditional among planted unsupported children and separate from parent support; all-child provenance is saved in case_metrics.csv.

Rare ratio is current rare mass / target rare mass. Original-rare retention tracks the original supported rare rows remaining rare. These can disagree strongly, especially in rare_hole. Cost targets are the original empirical real frequencies. Geometry targets and oracle labels are used only after controller actions.

## All 12 cost fixtures

P/F = pass/fail. Parent column is intended-mode agreement / support (%). Exact TV is candidate / E22. Rare is population ratio exact / native-four; retain is original-rare retention exact / native-four. Native end is unsupported % / TV. Gates are E/C1/C4/F1/F4.

| Case | Recall % | Parent % | Exact TV CB/E22 | Rare exact/full | Retain % exact/full | Native end bad%/TV | Ord/iso | CB gates | E22 gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nominal N1024/d8 | 100 | 69.6/100.0 | 0.0137/0.0000 | 1.000/1.000 | 100/100 | 0.00/0.0225 | 90/46 | F/F/F/F/F | P/P/P/P/F |
| nominal N2048/d8 | 100 | 60.9/100.0 | 0.0171/0.0000 | 1.000/1.000 | 100/100 | 0.00/0.0278 | 286/92 | F/F/F/F/F | P/P/P/P/F |
| nominal N4096/d8 | 100 | 66.3/100.0 | 0.0146/0.0000 | 1.000/1.000 | 100/100 | 0.00/0.0339 | 742/184 | F/F/F/F/F | P/P/P/P/F |
| nominal N8192/d8 | 100 | 72.1/100.0 | 0.0118/0.0000 | 1.000/1.061 | 100/100 | 0.00/0.0176 | 1636/369 | F/F/F/F/F | P/P/P/P/F |
| highdim N1024/d128 | 0 | —/— | 0.0449/0.0449 | 1.000/1.000 | 100/100 | 4.49/0.0449 | 0/0 | F/F/F/F/F | F/F/F/F/F |
| highdim N2048/d128 | 0 | —/— | 0.0449/0.0449 | 1.000/1.000 | 100/100 | 4.49/0.0640 | 88/0 | F/F/F/F/F | F/F/F/F/F |
| highdim N4096/d128 | 0 | —/— | 0.0449/0.0449 | 1.000/1.000 | 100/100 | 4.49/0.0449 | 0/0 | F/F/F/F/F | F/F/F/F/F |
| highdim N8192/d128 | 0 | —/— | 0.0450/0.0450 | 1.000/1.000 | 100/100 | 4.49/0.0450 | 28/0 | F/F/F/F/F | F/F/F/F/F |
| rare_hole N1024/d8 | 100 | 0.0/100.0 | 0.0449/0.0469 | 14.500/4.500 | 100/0 | 0.00/0.0557 | 71/46 | P/P/P/F/F | F/F/F/F/F |
| rare_hole N2048/d8 | 100 | 0.0/100.0 | 0.0444/0.0430 | 23.750/2.250 | 100/75 | 0.00/0.0337 | 257/92 | P/P/P/F/F | F/F/F/F/F |
| rare_hole N4096/d8 | 100 | 3.8/100.0 | 0.0427/0.0449 | 4.375/4.125 | 100/75 | 0.00/0.0374 | 760/184 | P/P/P/F/F | P/P/P/P/F |
| rare_hole N8192/d8 | 100 | 1.4/100.0 | 0.0443/0.0450 | 4.938/2.250 | 100/50 | 0.00/0.0338 | 1636/369 | P/P/P/F/F | P/P/P/P/F |

Nominal: detector precision/recall and parent support are 100%, but wrong-mode supported parents produce TV .01184–.01709 versus E22 TV 0, exceeding E22+.01 at all four N. This failure already exists in exact cloning. First-full support-repair precision is .719/.474/.474/.474 because ordinary moves enter the denominator; isolation-only precision is 1. Native-four nominal support remains 100%, with TV .01758–.03394 and rare ratio 1–1.061.

Highdim: every original isolation flag set is empty, leaving about 4.5% unsupported mass at every N. Ordinary moves sometimes occur but do not rescue the planted failures. A fast no-op cannot qualify.

Rare_hole: the broad original support/TV gates allow four isolation passes even though intended-mode agreement is 0–3.8% and rare mass is inflated 4.375–23.75×. Native ordinary transport reduces some inflation but loses original supported rare rows (native-four retention 0/75/75/50%) and full quality fails. Retaining the original gate is not an endorsement of these population allocations.

## All 28 geometry fixtures

Detector column is TP/FP/rare-FP with P/F for the original detector gate. Parent is realized/expected intended-mode validity (%). Quality cells are unsupported % / rare ratio / TV, with churn % appended for common/native. Gates are exact/common-four/native-four. Every E22 geometry verdict is F/F/F; matched numerical receipts appear below and in case_metrics.csv.

### trained600

| Case | Detector | CB parent % | CB exact | E22 parent % | E22 exact | CB common-four | CB native-four | Ord/iso | CB gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear N2048/d2 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 4.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 0.00/1.000/0.0000/0.0 | 0/85 | F/F/F |
| linear N2048/d16 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 4.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 0.00/1.000/0.0000/0.0 | 29/85 | F/F/F |
| linear N2048/d64 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 4.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 0.00/1.000/0.0000/0.0 | 0/85 | F/F/F |
| linear N2048/d128 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 4.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 0.00/1.000/0.0000/0.0 | 0/85 | F/F/F |
| fold N2048/d2 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 4.00/0.196/0.0400 | 0.44/0.912/0.0044/0.0 | 2.93/0.608/0.0376/0.0 | 37/85 | F/F/F |
| fold N2048/d16 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 4.00/0.196/0.0400 | 0.15/0.971/0.0015/0.0 | 0.63/0.882/0.0063/0.0 | 0/85 | F/F/F |
| fold N2048/d64 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 4.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 0.00/1.000/0.0000/0.0 | 0/85 | F/F/F |
| fold N2048/d128 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 4.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 0.00/1.000/0.0000/0.0 | 0/85 | F/F/F |
| fold_nuisance N2048/d2 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 4.00/0.196/0.0400 | 0.44/0.912/0.0044/0.0 | 2.93/0.608/0.0376/0.0 | 37/85 | F/F/F |
| fold_nuisance N2048/d16 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/1.3 | 0.73/0.176/0.0410 | 0.59/0.892/0.0059/0.0 | 0.63/0.882/0.0063/0.0 | 0/85 | F/F/F |
| fold_nuisance N2048/d64 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 1.5/0.8 | 0.68/0.186/0.0405 | 0.78/0.853/0.0078/0.0 | 0.00/1.000/0.0000/0.0 | 0/85 | F/F/F |
| fold_nuisance N2048/d128 | 82/3/1 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/1.0 | 0.73/0.176/0.0410 | 0.54/0.892/0.0054/0.0 | 0.00/1.000/0.0000/0.0 | 0/85 | F/F/F |
| fold_nuisance N1024/d128 | 0/0/0 F | —/— | 4.00/0.196/0.0400 | —/— | 4.00/0.196/0.0400 | 4.00/0.196/0.0400/0.0 | 4.00/0.196/0.0400/0.0 | 0/0 | F/F/F |
| fold_nuisance N4096/d128 | 164/4/0 P | 100.0/100.0 | 0.00/1.000/0.0000 | 1.0/0.6 | 1.73/0.205/0.0398 | 0.61/0.888/0.0061/0.0 | 0.00/1.000/0.0000/0.0 | 0/168 | P/F/P |

### frozen_initialization

| Case | Detector | CB parent % | CB exact | E22 parent % | E22 exact | CB common-four | CB native-four | Ord/iso | CB gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear N2048/d2 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 100.0/100.0 | 0.00/1.000/0.0000 | 0.00/1.000/0.0000/0.0 | 0.00/0.980/0.0010/0.0 | 156/86 | F/F/F |
| linear N2048/d16 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 100.0/100.0 | 0.00/1.000/0.0000 | 0.00/1.000/0.0000/0.0 | 0.00/1.176/0.0088/0.0 | 115/86 | F/F/F |
| linear N2048/d64 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 100.0/100.0 | 0.00/1.000/0.0000 | 0.00/1.000/0.0000/0.0 | 0.00/0.980/0.0010/0.0 | 105/86 | F/F/F |
| linear N2048/d128 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 100.0/100.0 | 0.00/1.000/0.0000 | 0.00/1.000/0.0000/0.0 | 0.00/1.176/0.0088/0.0 | 104/86 | F/F/F |
| fold N2048/d2 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 0.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 9.72/0.627/0.0972/0.0 | 273/86 | F/F/F |
| fold N2048/d16 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 0.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 1.42/0.990/0.0483/29.3 | 384/274 | F/F/F |
| fold N2048/d64 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 0.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 0.05/0.990/0.0029/0.0 | 314/86 | F/F/F |
| fold N2048/d128 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 0.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 0.00/1.088/0.0088/0.0 | 233/86 | F/F/F |
| fold_nuisance N2048/d2 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.0/0.0 | 0.00/0.196/0.0400 | 0.00/1.000/0.0000/0.0 | 9.72/0.627/0.0972/0.0 | 273/86 | F/F/F |
| fold_nuisance N2048/d16 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 1.2/1.5 | 0.00/0.196/0.0400 | 0.54/0.922/0.0054/0.0 | 1.42/0.990/0.0483/29.3 | 384/274 | F/F/F |
| fold_nuisance N2048/d64 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 3.7/0.7 | 0.00/0.216/0.0391 | 0.63/0.902/0.0063/0.0 | 0.05/0.990/0.0029/0.0 | 314/86 | F/F/F |
| fold_nuisance N2048/d128 | 82/4/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 1.2/1.1 | 0.00/0.196/0.0400 | 0.59/0.892/0.0059/0.0 | 0.00/1.088/0.0088/0.0 | 233/86 | F/F/F |
| fold_nuisance N1024/d128 | 41/0/0 P | 100.0/100.0 | 0.00/1.000/0.0000 | 2.4/1.2 | 0.00/0.216/0.0391 | 0.39/0.922/0.0039/0.0 | 0.00/0.961/0.0137/0.0 | 51/41 | P/P/P |
| fold_nuisance N4096/d128 | 164/11/0 F | 100.0/100.0 | 0.00/1.000/0.0000 | 0.6/0.6 | 0.00/0.205/0.0398 | 0.59/0.902/0.0059/0.0 | 0.05/1.127/0.0085/0.0 | 527/175 | F/F/F |

Trained600 N2048: all planted rows are detected and their realized/expected intended-mode parent validity is 100%, but 3 supported rows are flagged, including 1 supported rare row. The zero-rare-FP gate fails. N1024 nuisance has no flags; N4096 nuisance passes exact/native but fails common jitter rare retention (.888). No detector was repaired or bypassed.

Frozen initialization N2048: 4 supported false positives give legitimate FPR 4/1966=.002035, above .002. The critic change improves some mechanisms but does not relax this gate. N1024 nuisance is the sole complete geometry case pass; N4096 has 11 false positives and fails.

Native jitter introduces independent failures: trained folded d2 has 2.93% unsupported centers and rare ratio .608; frozen-init folded d2 has 9.72% unsupported centers and rare ratio .627. Frozen-init folded d16 has 1.42% unsupported, TV .04834, and planted repeated-write fraction .293. Exact parent correctness is therefore insufficient for native reaction quality. Common DV12 and native jitter outcomes remain distinct.

Full four-turnover parent support, weighted by actual moves:

| Family/critic | Backend | Ordinary parent support % | Isolation parent support % |
| --- | --- | --- | --- |
| cost:frozen_toy_head | cb64_ra | 99.96 | 100.00 |
| cost:frozen_toy_head | e22 | 100.00 | 96.69 |
| geometry:trained600 | cb64_ra | 100.00 | 100.00 |
| geometry:trained600 | e22 | 100.00 | 36.10 |
| geometry:frozen_initialization | cb64_ra | 100.00 | 100.00 |
| geometry:frozen_initialization | e22 | 100.00 | 100.00 |

## Native cost and work

Times include real FIFO ingestion, feature extraction, all reference setup/projection, count tests, support and both parent plans, native jitter/copying, EMA/optimizer/history actuation, refresh, and invalidation. They are first full turnovers measured once at each fixed N; no repeats or alternate seeds were selected. There are zero controller backward passes. This CPU toy timing is not a large-network throughput prediction.

| N | CB ms | E22 ms | Speedup | G rows CB/E22 | Critic rows CB/E22 | CB RSS after MiB | Process peak so far MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 40.421 | 51.825 | 1.28 | 2112/2048 | 3136/3072 | 603.85 | 600.53 |
| 2048 | 44.450 | 145.963 | 3.28 | 4290/4096 | 6338/6144 | 622.19 | 621.65 |
| 4096 | 74.285 | 533.757 | 7.19 | 8580/8192 | 12676/12288 | 623.66 | 677.37 |
| 8192 | 108.304 | 2218.330 | 20.48 | 17162/16384 | 25354/24576 | 638.59 | 768.98 |

Log-log time slopes: CB64-RA **0.500660**, E22 **1.812963**. The frozen speed criterion passes. CB forwards are 2N+moved generator rows and 3N+moved critic rows; E22 is 2N and 3N here. Reference costs include its ordinary reaction and native dense jitter. No candidate native cdist call occurs, but its manual projected pairwise products are explicitly counted:

| N | Projected distance entries | Projection multiply terms | Hypergeom terms | Parent entries | Max distance block | Snapshot retained bytes | E22 cdist pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 608640 | 2637824 | 1600 | 11776 | 256×256 | 91136 | 7526332 |
| 2048 | 1221504 | 5292544 | 3136 | 23552 | 256×256 | 168960 | 30105328 |
| 4096 | 2443008 | 10585088 | 6208 | 47104 | 256×256 | 324608 | 120421312 |
| 8192 | 4886464 | 21170688 | 12352 | 94464 | 256×256 | 635904 | 481717279 |

Separate exact-isolation phase timings demonstrate reference setup and target-search costs; they are included in that scope, not added again to native time:

| N | G/head features ms | Fit/setup ms | Support ms | Parent search ms |
| --- | --- | --- | --- | --- |
| 1024 | 1.171 | 14.848 | 0.739 | 3.933 |
| 2048 | 1.483 | 16.407 | 1.175 | 4.561 |
| 4096 | 3.205 | 26.001 | 2.991 | 7.484 |
| 8192 | 5.944 | 35.496 | 4.475 | 10.388 |

### Source cost and memory bounds

Let h be critic-head width, d latent width, o raw output width, r≤8, K≤64, B=256, A≤4×64=256 local parents, and m≤.05N isolation children. Four power passes cost O(Nhr+hr²); farthest-first initialization, four Lloyd passes, assignment/support, and cell pool scans cost O(NKr). Null/BH score sorting and priority/duplicate-row sorting cost O(N log N), with feature-coordinate comparison up to O(Nh log N). Exact conditional count enumeration is at most K(min(N/2,N)+1) terms. Parent search is O(mKr+mAr), plus bounded K/A sorting. Native jitter and row-state copying cost O((N+moved)d). Model forwards add O(N(C_G+C_D)) and moved-row refresh. With fixed h,d,o,K,r,A and fixed network size, favorable numerical work is linear in N and the worst sorting bound is N log N. Width/network growth adds the explicit factors; scaling K or r with N would invalidate this fixed-cap argument.

Source retains O(N(h+r+d+o)+mA+hr+Kr) data across learned features, projected caches, latents/EMA/optimizer/history, raw FIFO, and returned local-plan detail. The snapshot receipt alone excludes the raw FIFO and trainer/model tensors. Its largest observed retained array total is **643,584 bytes**; nominal N8192 is 635,904. The actual raw FIFO remains O(No), and feature arrays O(Nh). There is no N×N covariance: the basis is h×r. Candidate source evaluates chunk×K and chunk×A projected tables, never child×all-parent or latent-nearest-table search. A double 256×256 distance table is .5 MiB; one gathered 256×256×8 parent feature tensor is 4 MiB. Other live intermediates coexist, so these are table bounds, not a total-memory peak.

The measured process peak RSS reaches **1078.10 MiB** across this sequential run. /proc RSS after native CB calls ranges 603.85–791.59 MiB. Peak RSS is cumulative across both packages, setup, and earlier common dense diagnostics; allocator retention and differing kernel memory accounting prevent attributing it to the candidate. No isolated candidate peak-RSS claim is made.

The common DV12 diagnostic deliberately materializes child×all-table latent distances, O(mNd) work with chunked/dense memory. Its cost is saved separately in common_jitter.rounds[].jitter_cost and excluded from the native bounded-source claim. Zero cdist calls alone is not evidence of zero pairwise matmul work; the native table/work/source receipts above are the evidence.

## Six separate latent-sampling probes

These were explicitly authorized and preregistered after the original 80 rows, before executing the six probes. Same frozen family seeds, N2048, actual candidate support/parent selection and perturb_latent. Exactly repaired isolation centers are perturbed once across every row; no ordinary moves or retries are added. No gate or original verdict changes. They diagnose latent jitter only; output-noise support is undefined.

| Fixture | Clean support % | Latent-jitter support % | Rare ratio | TV | Mean norm | Max norm |
| --- | --- | --- | --- | --- | --- | --- |
| cost:nominal/d8 | 100.00 | 100.00 | 1.0000 | 0.0171 | 0.04891 | 0.05000000 |
| cost:highdim/d128 | 95.51 | 95.51 | 1.0000 | 0.0449 | 0.05000 | 0.05000000 |
| geometry:linear/d2 | 100.00 | 100.00 | 1.0000 | 0.0000 | 0.02975 | 0.05000011 |
| geometry:linear/d128 | 100.00 | 100.00 | 1.0000 | 0.0000 | 0.05000 | 0.05000003 |
| geometry:fold/d2 | 100.00 | 46.14 | 0.5196 | 0.5386 | 0.02975 | 0.05000011 |
| geometry:fold/d128 | 100.00 | 99.32 | 1.0000 | 0.0068 | 0.05000 | 0.05000003 |

Folded d2 support drops from 100% to **46.14%**, rare ratio to .520, and TV to .53857. Folded d128 retains 99.32% support. The fixed norm cap distributes perturbation across all latent coordinates; this generator depends mainly on two coordinates, so the two cases expose different effective output perturbations. Norm cap .05 is respected within float rounding (largest observed .050000114). No dimension- or oracle-adapted jitter was introduced.

## Integrity, reproduction, and recommendation

Only features, model/latent state, and private RNG streams enter the actual controllers. Raw geometry, intended modes, support and target masses enter evaluation after actions. E22 and CB start from identical latent/model hashes for all 40 pairs. Networks are held fixed. Actual ordinary plans and _move run with unforced controller state, including row-state copying and refresh; bounded static FIFO replay is not fresh independent real data or full GAN training. Count-test validity is conditional on the fixed partition and iid draws; repeated adaptive snapshots have no cumulative guarantee.

READY per-file package/config hashes were checked before import and after execution. Protocol, runner, models/fixture sources, package and config remain unchanged. Protocol SHA256: `afa81aff395ee9d37b775c3f5f8afde6a27183d82576ce3c0dd997bda15430b9`. Runner: `b03b895eda47ec43fad8f8a40fb111f731459ca0bbcd4c73d499e7ed1e7082c7`. Config: `d2b1018854671ded2ffe92b2f30c97a3871cf15911c6c241ceaadd281d94e7e7`. READY: `0d8f674fa2fac7b736b592e472310ca892037d4acf9e1de6d9f6a03e55710ac2`. Original hashes and audit are in integrity.json; report postprocessing preserves raw files and is receipted in analysis_receipt.json and artifact_manifest.json.

Commands from this directory:

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /tmp/pr38-default-env/bin/python -u run_validation.py --ready /ml2/hypergan/gan-attempts/feature-cells-config-20260929/implementation/READY.json > run.log 2>&1
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /tmp/pr38-default-env/bin/python -u sampling_probe.py > sampling.log 2>&1
/tmp/pr38-default-env/bin/python analyze_results.py
```

Artifacts: results.json retains all 80 raw rows, actions and candidate masks; results.csv is the original flat receipt; case_metrics.csv adds complete derived metrics and population vectors; analysis_results.json contains leaderboard/scaling accounting. sampling_results.json/csv contain only the six probes. Protocols, source, commands, logs and integrity receipts are preserved.

**Recommendation:** retain CB64-RA as an experimental alternative with a verified fixed-cap native implementation and improved geometry-aware center parent selection. Do not describe this config as passing the original portability/geometry requirements. Mass allocation, highdim detection and the configured served latent law remain demonstrated failures. Further work requires a new hypothesis/protocol; this lane stops after the authorized bounded study.
