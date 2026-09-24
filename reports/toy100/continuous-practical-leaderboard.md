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
