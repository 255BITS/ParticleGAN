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

The following board is preserved historical evidence, not the current assignment.

# Constant-LR GAN leaderboard

**Goal: make the GAN formulation itself acquire the fixed target and remain stable under continued training. No candidate is qualified.** Whole-target shifts are separate. Rest is acceptable when there is no learning signal; an error must still produce a corrective response.

The generator must learn from the discriminator's adversarial objective. Running a discriminator beside a separate density/coverage fitter does not satisfy this goal. We keep likelihood, anchor fitting and related geometric projections as references, not entries competing for the GAN solution.

Rank **acquisition first, then sustained quality and recovery**, with runtime and changes to the GAN rule visible. Original strict failures remain recorded. A temporary dip with measured recovery is not permanent collapse, and a perfect hold of an incomplete solution is not full acquisition. There is no defensible percent-to-solved.

| GAN candidate | Acquire: trajectory / ring | Continued same-target evidence | Practical judgment |
| --- | --- | --- | --- |
| **1. PR84: smoothed generator critic + alternating own-curvature bounds** | Trajectory PASS, MSE `.000942662`. Submitter reports eight-mode terminal suffix; our pinned cold reproductions end **seven/HQ1**, FAIL. | Our warm200 passes; dense borrowed hold1111/1200, temporary mode/HQ departures, final quality recovers. External hold114/120. No locally qualified full-eight own-acquired hold. | **Working reference.** About31–33 seconds for1200 cold updates on one thread. Adversarial G signal retained; acquisition and continuation remain unresolved. [PR84](https://github.com/255BITS/ParticleGAN/pull/84), [fresh independent control](pr93-cold-independent-audit.md). |
| **2. PR82: alternating own-curvature** | Trajectory PASS; independently reproduced cold ring **six/HQ `.892`**, FAIL. | Warm200 passes; no full-ring own-acquired hold. | Useful update-order correction; cannot acquire the whole ring. [Independent audit](pr82-independent-audit.md). |
| **3. Bounded critic refinement** | Trajectory PASS `.000910`; ring **three/HQ `.451`**, FAIL. | Saved44, warm200 and borrowed dense hold1200 pass. | Extra D fitting repairs stationary guidance but fails acquisition. Cold ring **789s**,69,328 inner-gradient attempts: impractical as the winner. [Source-bound cold result](pr84-critic-refinement-finite-cold.md). |
| **Rejected update: actual16-bank Adam means** | Cold withheld after short filter. | Repairs one bad step but only24/44 own consecutive saved-window checks pass. | Noise reduction alone fails this tested rule. [Native Adam evidence](mean16-continuous-screen.md). |
| **Rejected update: nonlocal moves selected by original Rp G loss** | Copied cold3→4 with converged actual neural fit; no full cold run. | Two exact native eight-update warm branches lose modes; all16 selected fits converge and lower G loss below both rest and native Adam. | **G-loss descent against the current critic is insufficient.** At1327 it deletes a mode's sole particle.4.69s for paired original/candidate branches. [Native failure](pr84-adversarial-native16.md). |
| **Rejected update: fixed common instance noise** | One-step cold directions point toward an empty mode, but copied alternating eight-update continuation ends one mode/HQ `.152`. | Copied warm continuation ends four modes/HQ `.268` after eight G updates. | Fixed-width observation channel fails this test. Copied fitted critic retains old D Adam moments; not a source-exact native qualification. [Failure and limits](common-instance-noise-round10.md). |
| **Diagnostic: profiled sharp critic value** | Copied cold state stays at three modes after six certified improving moves. | Copied warm transitions improve quality; no live full training pass. | Still adversarial, but frozen features/metric and expensive inner fitting. Useful causal evidence, not a practical acquisition candidate. [Bounded continuation](pr84-convex-profiled-cold-continuation.md). |

PR84's reported ring pass does not reproduce under our pinned PyTorch2.13.0+cu126 CPU/AVX2 environment. We preserve both outcomes; we do not average them or run seed sweeps to choose the favorable one. Sparse external holds and dense local holds are labeled separately. The [source snapshot](continuous-evidence/practical-board/external-pr-status.json) pins submitted PR heads and claims.

The next single bet must address **both** the cold acquisition barrier and harmful stationary guidance using a discriminator-driven update. Both recent continuations above failed, despite encouraging one-step results. The bounded next diagnostic compares the discriminator's response to a destructive warm1327 move and a useful cold472 move: four fixed-feature readout fits, shared banks within each pair, value bounds and crossed G-loss comparisons. It is a diagnostic, not a training candidate or a neural acquisition result. See [current mechanisms and research](gan-dynamics-current-bets.md). Extra radius/rho cuts, weaker gates, zero-centered pulls, and standalone fitting objectives are not the plan.

Reference results that help explain the problem, without qualifying as the GAN solution:

- **Sampled anchors:** neural eight-mode acquisition, own hold1200, and error repair pass. A conditioned missing batch loses a mode for two updates then recovers. The longer run was deliberately stopped on this scope clarification at9488, with7088/7088 resumed coverage checks passing; **not a12000-step pass**. It uses a separate coverage objective. Particle counts `[1,1,1,1,1,1,1,5]` and spread `.029` versus target `.07` show substantial fidelity error. [Fidelity audit](sample-anchor-endpoint-fidelity-audit.md), [reference board](continuous-neural-reference-board.md).
- **Likelihood donor/EM:** six bank stress gates and44/44 native neural saved-window checks pass. Its replacement G move is chosen without the discriminator; it is smoothed density fitting, currently limited to the small2D host and growing data history. Full neural acquisition is untested. [Neural filter and cost](forward-kl-neural44.md).
- **PR88/89:** reported acquisition contenders, but their extra updates are accepted using a geometric target criterion. [PR88's independent cold run](pr88-cold-independent-audit.md) ends eight/HQ `.880`, missing the terminal quality gate; PR89's critic-selected centroid pull does not check adversarial loss improvement. They are references for proposal mechanisms, not evidence that the GAN game is stable.
- **PR81:** target-error cap, trajectory near-miss `.02110`, no ring pass. **PR93/94:** support constraints obtain strong borrowed hold but fail acquisition (independentPR93 four modes; reportedPR94 six). **PR85/86/87:** warm failures. **PR90/92:** diagnostics. **PR91/96:** preservation variants without full acquisition; PR96's single observedHQ dip is not proof of lasting collapse.

Production PR60 head remains `983d037a7028afc5c1b0df4d4097eff8e6abe9b5`, with scheduled LR decay. No production22 pass, merge, or unavoidable no-go theorem is claimed. The remaining production gates include full mass/shape accuracy and practical cost, after a GAN candidate passes acquisition and its own continued same-target training.
