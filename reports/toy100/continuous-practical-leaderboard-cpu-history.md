# Stability after confirmed convergence

[PR140/PR143 reproduction audit](pr140-pr143-repro-audit/README.md): MKL CPU-vendor
dispatch explains the previous discrepancy. With the diagnostic dispatch
override, all 2,400 update records and 240 quality checks match each submission
exactly. PR140 reproduces114/120; PR143 reproduces115/120, minimum seven modes,
five single-check misses. PR143 is a **provisional keep / best sparse stay in
this compared family**. Ordinary AMD runs still fail acquisition. These sparse
quality-triggered-policy results are separate from the dense hold below;
neither PR is release-qualified or a confirmed global winner.

The current objective separates learning time from stability once quality has
converged. The first200 consecutive8-mode/HQ>=.90 observations confirm convergence;
the following1200 updates are a disjoint hold. Early dips do not fail that hold.
See [criterion](h_stability/CONVERGENCE.md). All original toy results remain intact.

| Candidate | Confirmed at update | Passing hold checks before failure | Original toy coverage |
| --- | ---: | ---: | --- |
| eps_net_1m | 1400 | 291/1200 | 10/19 PASS |
| Shared column RMS | 2404 | 120/1200 | two_pole/ring/unipolar PASS; other gates incomplete |
| Shared RMS | 3241 | 95/1200 | two_pole/ring/unipolar PASS; other gates incomplete |
| H | Not confirmed by6000 | Not started | 13/19 PASS |

No candidate passes the post-convergence hold. The epsilon base remains selected.
[Baseline evidence](h_stability/convergence-baseline/summary.json),
[comparisons](h_stability/convergence-comparisons/results.json). The earlier
491/83/82/54 counts below measure the old hold beginning at1200, not this criterion.

## Latest selected-base audit

`eps_net_1m` improves own-state stability to491 consecutive passing checks, then
fails at update1692 (HQ0.864746094). Its complete older-host audit is10/19 PASS;
it is not an overall toy winner. See [current results](h_stability/RESULTS.md).

# Current experiment starting point (2026-09-24)

**eps_net_1m** is the selected research base. The GAN objective is unchanged;
G/D Adam epsilon is .001, particle epsilon1e-8, and learning rates remain fixed.
It is not a release-qualified or overall leaderboard winner.

| Candidate | Cold ring | Own-state continuation | Known coverage limit |
| --- | --- | --- | --- |
| Selected eps_net_1m | PASS,8 modes/HQ.996826,suffix14 | Short200 PASS; long hold FAIL at1692 after491 passes | older10/19 PASS; native3 SKIPPED |
| g_threequarter_rate | PASS,8 modes/HQ.999756,suffix9 | FAIL at1284 after83 passing checks | two_pole FAIL; other17 UNRUN |
| H acquisition control | PASS,8 modes/HQ.999268,suffix5 | FAIL at1255 after54 passing checks | older13/19 PASS; full hold FAIL |

Each own-state result starts from that candidate's own acquired optimizer/model/
RNG state. No candidate's toy passes are credited to another. See
[current recipe](h_stability/current-base.json) and [replay guide](h_stability/START.md).

Historical comparison reports remain on the [research branch](https://github.com/255BITS/ParticleGAN/blob/14a7818157f8a4254dcbb3a40a562abc93e58402/reports/toy100/continuous-practical-leaderboard.md).
