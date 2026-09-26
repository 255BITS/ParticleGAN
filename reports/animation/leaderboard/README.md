# Animation world model leaderboard

Primary score: test dream score, the mean standardized-state MSE against the exact simulator at dream horizons 1, 5, 20 and 50, rolled out closed-loop from 3 start offsets (0, 15, 30) in every test episode. Each state's error is capped at 10 so one divergent rollout cannot dominate; Diverged is the fraction of rollouts at the cap at step 50. Lower is better; the exact simulator scores 0. First fail is the median first step whose standardized-state MSE exceeds 1. One-step MSE is teacher-forced over every test transition. Frame self compares dreamed frames with renders of the dreamed states (self-consistency); frame true compares them with renders of the simulator states.

Every trained arm uses 20,000 updates, batch 256, lr 0.001 and training seed 24002; GAN rows use MoG1024 unless the row says otherwise (the direct arm has no prior). Persistence is untrained. No seed-only repeats.

Rows are refused unless dataset hashes, the evaluation/simulator source hashes and the training budget fields match. Checkpoint selection is minimum validation dream score.

| Rank | Run | Arm | Dream score ↓ | 1 | 5 | 20 | 50 | First fail (median step) | OOD dream score ↓ | OOD/ID ratio | OOD diverged | One-step ↓ | Frame self ↓ | Frame true ↓ | G / E / D params | Train s |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | direct | direct | 0.13596 | 0.00229 | 0.02594 | 0.15500 | 0.36060 | 35 | 2.36349 | 17.38 | 32.2% | 0.00262 | 0.00049 | 0.05102 | 891,495 / 0 / 0 | 412.2 |
| 2 | enc_w4 | gan | 0.28041 | 0.00526 | 0.09121 | 0.44036 | 0.58479 | 19 | 1.58518 | 5.65 | 0.0% | 0.00549 | 0.00459 | 0.08887 | 1,031,277 / 149,824 / 1,531,075 | 6624.0 |
| 3 | base | gan | 0.31075 | 0.01149 | 0.06738 | 0.50401 | 0.66013 | 21 | 1.91263 | 6.15 | 0.3% | 0.00852 | 0.00404 | 0.09114 | 1,031,277 / 149,824 / 1,531,075 | 5877.1 |
| 4 | route_temp1 | gan | 0.35964 | 0.02039 | 0.09674 | 0.56338 | 0.75807 | 19 | 2.00086 | 5.56 | 0.0% | 0.01850 | 0.00424 | 0.10140 | 1,031,277 / 149,824 / 1,531,075 | 6626.0 |
| 5 | mog64 | gan | 0.36828 | 0.01855 | 0.10514 | 0.54469 | 0.80476 | 20 | 1.68774 | 4.58 | 0.0% | 0.01772 | 0.00473 | 0.10421 | 1,031,277 / 149,824 / 1,531,075 | 6619.5 |
| 6 | detach | gan | 0.47241 | 0.02143 | 0.15118 | 0.75773 | 0.95929 | 18 | 1.98210 | 4.20 | 0.3% | 0.01784 | 0.00773 | 0.11728 | 1,031,277 / 149,824 / 1,531,075 | 5876.5 |
| 7 | synth_w4 | gan | 0.66861 | 0.05315 | 0.30047 | 1.00885 | 1.31198 | 15 | 2.17471 | 3.25 | 0.0% | 0.04396 | 0.13810 | 0.14986 | 1,031,277 / 149,824 / 1,531,075 | 6620.0 |
| 8 | detach_no_g1_g3_anchor | gan | 0.73519 | 0.04895 | 0.35975 | 1.19430 | 1.33778 | 12 | 2.17947 | 2.96 | 0.0% | 0.04044 | 0.14278 | 0.15220 | 1,031,277 / 149,824 / 1,531,075 | 5875.5 |
| 9 | detach_no_joint_d | gan | 0.86204 | 0.05117 | 0.35642 | 1.09528 | 1.94531 | 14 | 2.35745 | 2.73 | 0.0% | 0.04421 | 0.10639 | 0.12814 | 1,031,277 / 149,824 / 707,042 | 3923.3 |
| 10 | persistence | persistence | 1.26363 | 0.16005 | 0.85692 | 2.34210 | 1.69544 | 8 | 3.87764 | 3.07 | 5.0% | 0.13806 | 0.00000 | 0.13674 | 0 / 0 / 0 | 0.0 |

Simulator floor: **0** on every dream metric. G params for the direct arm are its predictor.

## Findings

**Summary.** Every GAN row beats persistence by a wide margin in distribution (20-step 0.50–1.19 vs 2.34). The supervised `direct` baseline still wins in distribution: its dream score is 0.136 vs 0.311 for the best GAN row. Out of distribution the ranking flips. The GAN rows have the best OOD dream scores (1.91 / 1.98 vs 2.36 for direct), and only 0.3% of their rollouts diverge, vs 32.2% for direct. The error cap hides how far direct's OOD rollouts actually go (unbounded, around 1e9 in calibration). The GAN dream stays on a plausible sprite manifold when it meets states it never saw; the regression extrapolates off it. No row learns the unseen ceiling bounce, so OOD first failure comes at step 3–8 for every learned model, and 14 for direct.

**Baseline promotion (2026-09-26).** The live-composition row (formerly `no_detach`) is now `base`, and the detached original is `detach`. The two older ablations were run with the detach on, so they are relabeled `detach_no_joint_d` and `detach_no_g1_g3_anchor` and compare against `detach`. Result directories were renamed rather than rerun; each summary records `relabeled_from`.

**Ablations** (each changes one switch from `detach`):
- **Joint critic matters.** Removing it (`detach_no_joint_d`) moves the dream score from 0.472 to 0.862. Prior samples stop obeying the dynamics (prior dynamics MSE 0.046 → 0.118), and G3 stops matching G1 (prior frame MSE 0.019 → 0.171).
- **Anchoring E on G1 and G3 as well as G2 matters.** Training E through G2 only (`detach_no_g1_g3_anchor`) raises the dream score to 0.735. Frame self-consistency collapses: 0.143 vs 0.008, meaning dreamed frames no longer match the dreamed state. E lands z off the region G3 was trained on, as predicted.
- **The detach hurts here.** `base` (live) beats `detach` in distribution (0.311 vs 0.472) and out (1.913 vs 1.982). The lunar feedback-loop fix is not needed on this problem; the live E input improves composition.

**Failed checks.** Encoder routing uses only 2–7 of 1,024 mixture components (2.0–2.9 effective). The model behaves like a continuous autoencoder around a handful of centers, well short of the plan's "more than half in use" health check. 10–39% of offset coordinates sit at their bound; `detach_no_g1_g3_anchor` is highest at 38.6%. GAN rows trained four at a time on one GPU, so their train seconds are inflated relative to `direct`.

**Iteration 1** (one change each against `base`; target: beat 0.311 in distribution and 1.913 OOD):

| Candidate | Change | Dream ↓ | OOD dream ↓ | Components used | Verdict |
|---|---|---:|---:|---:|---|
| `enc_w4` | real encoding weight 1 → 4 | **0.280** | **1.585** | 3 | Beats base on both |
| `mog64` | MoG1024 → MoG64 | 0.368 | 1.688 | 1 | Better OOD, worse ID |
| `route_temp1` | routing temperature .25 → 1 | 0.360 | 2.001 | 19 | Spreads routing, worse on both |
| `synth_w4` | synthetic weight 1 → 4 | 0.669 | 2.175 | 3 | Much worse on both |

- **`enc_w4` wins.** It is the only candidate that beats base in and out of distribution: 10% better in distribution and 17% better OOD, with no diverged OOD rollouts. One-step error drops from 0.0085 to 0.0055. The cost is looser prior samples (prior dynamics MSE 0.010 → 0.060): the extra anchor pulls capacity toward the encoded path. It is still 2.1× behind `direct` in distribution.
- **Routing is not what drives quality.** `mog64` routes every state to a single component and still has the second-best OOD score. `route_temp1` is the only run that uses more components (19 used, 10.8 effective), but it scores worse. In this setup the encoder acts as a continuous code around one or a few centers, and the mixture adds little. The "more than half of components used" health check is the wrong target here.
- **Weighting synthetic reconstruction up hurts.** `synth_w4` breaks frame self-consistency (0.004 → 0.138, about the level of the no-anchor ablation). Its validation score peaked at 5k steps and then got worse.
- **Base, `enc_w4` and `mog64` were still improving at 15–20k steps; `route_temp1` and `synth_w4` peaked early (10k and 5k).** All four ran together on cuda:1, so their training time is about four times what one run takes alone.

**Recommendations** (iteration 2, one change each against `enc_w4` as the new base):
1. Promote `enc_w4` to `base`.
2. Push the anchor further with `enc_w8` (real encoding weight 8), watching prior validity.
3. Combine the two OOD winners: `enc_w4` plus MoG64 (`enc_w4_mog64`).
4. Make runs faster before iterating more: apply the gradient penalty lazily, do the EMA update as one fused call, and run at most two jobs per GPU. Every row has to be rerun after this.
5. Still open: an OOD-trained `direct` reference as the OOD upper bound.

## OOD animation

ood_test starts higher (y0 ∈ [0.75, 0.9]) with speeds 1.2–1.6 launched upward, so the sprite hits the ceiling, which never happens in training (every in-distribution state stays below y = 0.75). 24.0% of all OOD states and 38.2% of states in the first 50 steps fall outside the per-coordinate training box; 90.5% of OOD episodes bounce off the ceiling within 50 steps.

| Run | OOD dream score ↓ | 1 | 5 | 20 | 50 | First fail (median step) | Diverged | Frame self ↓ | Frame true ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 2.36349 | 0.17281 | 1.00891 | 3.93204 | 4.34020 | 14 | 32.2% | 0.14040 | 0.22509 |
| enc_w4 | 1.58518 | 0.38111 | 0.90985 | 2.79962 | 2.25012 | 9 | 0.0% | 0.01412 | 0.14954 |
| base | 1.91263 | 0.44041 | 1.08263 | 3.49072 | 2.63677 | 8 | 0.3% | 0.01737 | 0.15387 |
| route_temp1 | 2.00086 | 0.64906 | 1.27419 | 3.54112 | 2.53906 | 6 | 0.0% | 0.01160 | 0.15979 |
| mog64 | 1.68774 | 0.37683 | 1.01506 | 3.07702 | 2.28205 | 7 | 0.0% | 0.01102 | 0.15685 |
| detach | 1.98210 | 0.46919 | 1.14534 | 3.80912 | 2.50476 | 6 | 0.3% | 0.02020 | 0.15868 |
| synth_w4 | 2.17471 | 0.70768 | 1.59721 | 3.60816 | 2.78581 | 4 | 0.0% | 0.13734 | 0.15687 |
| detach_no_g1_g3_anchor | 2.17947 | 0.69519 | 1.59469 | 3.74282 | 2.68517 | 4 | 0.0% | 0.14887 | 0.16925 |
| detach_no_joint_d | 2.35745 | 0.82858 | 1.70729 | 3.89358 | 3.00036 | 3 | 0.0% | 0.10363 | 0.15456 |
| persistence | 3.87764 | 0.57712 | 2.68203 | 7.11214 | 5.13927 | 4 | 5.0% | 0.00000 | 0.16219 |

## Prior and encoder health

Prior dynamics/frame MSE score G2 and G3 on prior samples against the exact step and render of the sampled G1 state. Encoder health is measured on every test state.

| Run | Prior dynamics MSE ↓ | Prior frame MSE ↓ | Offset at bound ↓ | Components used | Effective components |
|---|---:|---:|---:|---:|---:|
| enc_w4 | 0.06031 | 0.02855 | 0.062 | 3 | 2.8 |
| base | 0.01008 | 0.01191 | 0.102 | 3 | 2.9 |
| route_temp1 | 0.00701 | 0.00840 | 0.128 | 19 | 10.8 |
| mog64 | 0.02919 | 0.02022 | 0.078 | 1 | 1.0 |
| detach | 0.04606 | 0.01870 | 0.117 | 3 | 2.2 |
| synth_w4 | 0.08308 | 0.15943 | 0.132 | 3 | 2.9 |
| detach_no_g1_g3_anchor | 0.51559 | 0.16840 | 0.386 | 7 | 2.7 |
| detach_no_joint_d | 0.11836 | 0.17067 | 0.127 | 4 | 2.0 |

## Runs

- **direct:** Supervised st -> (st+1, gt) baseline with the same frame decoder, MSE on next state and frame; no prior, encoder or critics. [Config](direct_config.yaml) · [Dream](direct_dream.gif)
- **enc_w4:** GAN world model: G1/G2/G3 from the MoG prior, E(st) -> z dream loop; joint + marginal critics; E anchored on st/st+1/gt; live synthetic composition; real encoding weight 4. [Config](enc_w4_config.yaml) · [Dream](enc_w4_dream.gif)
- **base:** GAN world model: G1/G2/G3 from the MoG prior, E(st) -> z dream loop; joint + marginal critics; E anchored on st/st+1/gt; live synthetic composition. [Config](base_config.yaml) · [Dream](base_dream.gif)
- **route_temp1:** GAN world model: G1/G2/G3 from the MoG prior, E(st) -> z dream loop; joint + marginal critics; E anchored on st/st+1/gt; live synthetic composition; routing temperature 1. [Config](route_temp1_config.yaml) · [Dream](route_temp1_dream.gif)
- **mog64:** GAN world model: G1/G2/G3 from the MoG prior, E(st) -> z dream loop; joint + marginal critics; E anchored on st/st+1/gt; live synthetic composition; MoG64 prior. [Config](mog64_config.yaml) · [Dream](mog64_dream.gif)
- **detach:** GAN world model: G1/G2/G3 from the MoG prior, E(st) -> z dream loop; joint + marginal critics; E anchored on st/st+1/gt; detached synthetic composition. [Config](detach_config.yaml) · [Dream](detach_dream.gif)
- **synth_w4:** GAN world model: G1/G2/G3 from the MoG prior, E(st) -> z dream loop; joint + marginal critics; E anchored on st/st+1/gt; live synthetic composition; synthetic reconstruction weight 4. [Config](synth_w4_config.yaml) · [Dream](synth_w4_dream.gif)
- **detach_no_g1_g3_anchor:** GAN world model: G1/G2/G3 from the MoG prior, E(st) -> z dream loop; joint + marginal critics; E anchored on st+1 only (no G1/G3 anchor); detached synthetic composition. [Config](detach_no_g1_g3_anchor_config.yaml) · [Dream](detach_no_g1_g3_anchor_dream.gif)
- **detach_no_joint_d:** GAN world model: G1/G2/G3 from the MoG prior, E(st) -> z dream loop; marginal critics only (no joint D); E anchored on st/st+1/gt; detached synthetic composition. [Config](detach_no_joint_d_config.yaml) · [Dream](detach_no_joint_d_dream.gif)
- **persistence:** No training: st+1 = st with the exact render of st. The baseline any useful model must beat. [Config](persistence_config.yaml)
