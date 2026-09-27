# Sparse-UCD: simple-critic transfer (seed 1, 5000 steps, constant LRs, no instance noise)

Ranked by: bar held at end, then evals on bar (of 50), then modes, then joint*hq. `held` = evals on bar / evals since first crossing. D diagnostics: max |d(real)[c]| over every training step; max ||grad D|| over real/fake/interp at the 50 eval steps; final D gap = mean d(r) - d(f); gn_i = final median ||grad D|| on interpolates.

| # | arm | formulation | bar | bar_step | held | on_bar | modes | min modes 2nd half | hq | cond | sep | sym | joint | sp@1e-2 | sp@1e-3 | zero | core | w1 | ucdF | max abs D(real) | max grad-norm | final D gap | gn_i | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | champ_ref_anneal | REF: champion s1 as archived (RpGAN + g_interp_cap(1), cosine LR anneal from 60% to 5%) | fail | n/a | 0/0 | 0 | 63 | 54 | 0.975 | 0.986 | 1.02 | 0.986 | 0.986 | 0.972 | 0.950 | 0.95 | 0.74 | 0.037 | 1.00 | n/a | n/a | n/a | n/a | 36.7 |
| 1 | champ_matched | RpGAN + g_interp_cap(1) [constant LRs] | fail | n/a | 0/0 | 0 | 62 | 30 | 0.985 | 0.997 | 1.06 | 0.997 | 0.997 | 0.961 | 0.935 | 0.93 | 0.41 | 0.035 | 1.00 | 4.95 | 1.96 | 0.29 | 0.81 | 81.3 |
| 2 | sec_nodamp | wgan + r1(1) + secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] (= secant_r1_b2 here: no A2 damping in this harness) | fail | n/a | 0/0 | 0 | 58 | 32 | 0.884 | 0.968 | 1.04 | 0.968 | 0.967 | 0.964 | 0.955 | 0.95 | 0.74 | 0.049 | 0.78 | 5.41 | 2.62 | 0.59 | 0.75 | 54.7 |
| 3 | sec_lazy4 | wgan + r1(1) + secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, lazy_k=4] | fail | n/a | 0/0 | 0 | 54 | 21 | 0.755 | 0.978 | 1.04 | 0.976 | 0.975 | 0.962 | 0.955 | 0.95 | 0.85 | 0.065 | 0.71 | 5.51 | 2.44 | 0.79 | 0.86 | 79.9 |
| 4 | sec_rpbase | RpGAN + r1(1) + secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] | fail | n/a | 0/0 | 0 | 45 | 19 | 0.801 | 0.971 | 1.15 | 0.971 | 0.970 | 0.984 | 0.981 | 0.98 | 1.01 | 0.128 | 0.93 | 4.08 | 1.92 | 0.89 | 0.71 | 52.7 |

## Protocol

- Worktree/branch `sparse-ucd-secant` (from `sparse-ucd` @ 4be4a5ab). Launcher `experiments/sparse_secant.sh`, configs `configs/sparse-secant/*.yaml`, board `experiments/sparse_secant_board.py`. Tail with `tail -f results/sparse-secant/*.log` (one line per eval: study metrics, then `Dr`, `gap`, `|Dr|max`, grad-norm med at r/f/i, max, LR scale).
- Champion config `champion/l0p02_gw_sp0p003` (emb_dim 0, prior_partition class, hidden 256, z 16, Fourier ramp 0.3-0.7, ucd_lambda 0.02, gated head from 40%, lambda_sp 0.003), seed 1, 5000 steps, batch 256, GPU cuda:1.
- Instance noise: the harness has none (nothing to disable). LR anneal: the harness has a delayed cosine (60% to a 5% floor); every arm here sets `lr_anneal_start: 1.0`, so LRs are constant (the `lr` column in the logs stays 1.00). Fourier ramp, gate warm start and gumbel symbol sampling are part of the recipe, not noise or LR schedules, and are kept.
- New regularizer arm `h_secant_cap` in `lib/grad_regularizers.py`: `lam_r1 E||g(r)||^2 + lam_path E relu(t|r_nn - f| - (d(r_nn)[c] - d(f)[c]))^2 + lam_cap E_{r,f,i} relu(||g|| - 1)^2`, on D's joint input [x | y] at the requested class c, r_nn = nearest real of the same class in the batch, i = f + u(r - f), u ~ U[0.1, 0.9]. New keys (defaults reproduce the study): `d_base` (gan/wgan), `d_beta2`, `lazy_k`, `lam_r1`, `lam_path`, `path_target`, `lam_cap`, `path_u`. D diagnostics use their own RNG stream: a 500-step replay of champion s1 with diagnostics on matched the archived `metrics.jsonl` exactly at every eval.
- `secant_r1_b2` and `sec_nodamp` differ only in A2 latent damping, which this harness does not have (the prior takes a plain Adam step). They are the same formulation here, so it ran once (`sec_nodamp`). The two freed slots went to distinct formulations: `sec_rpbase` (same terms on the harness's RpGAN D loss) and `sec_lazy4` (lazy_k 4).

## Findings

1. **No arm reaches the bar.** The archived champion s1 did not either (63/64). `champ_matched` is best at the end: 62/64, hq 0.985, cond/sym/joint 0.997. It is unstable without the anneal, though: modes fall to 30 at step 3500 (the archived annealed run never went below 54 in its second half), and its core width collapses to 0.41 (0.74 with the anneal).
2. **The ring candidate does not transfer.** `sec_nodamp` ends at 58/64, hq 0.884, joint 0.967 and w1 0.049, below `champ_matched` on every coverage metric. It does keep a healthier core width (0.74) and 95% exact zeros. D's class head is less sure on fakes (ucdF 0.78 vs 1.00). It also runs about 1.5x slower (55 vs 81 steps/s).
3. **The secant term dominates the critic loss and is never satisfied.** At the end t_path is 0.64 while cap is 0.13 and r1 0.07, and 82% of fakes still violate `t*|r_nn - f|` (mean nearest-real distance 0.42). D ends flat at the samples (grad-norm median 0.22 at real and 0.26 at fake) and steep only in between (0.75). In 24-D with 3-sparse modes, a same-class nearest real is often a different mode, so the secant pulls fakes toward a mode they should not join.
4. **D is already tame in this harness.** Max |d(real)| is 4 to 5.5 and max grad-norm is 1.9 to 2.6 in every arm, the champion included. The ring study's main gain, stopping D spikes, has nothing to fix here.
5. **The RpGAN base is worse** (`sec_rpbase` 45/64, w1 0.128): the secant fights the saturating logistic. **lazy_k 4 is worse** (54/64, hq 0.755), the same as in the ring study, although it recovers the speed (80 steps/s).
6. Every arm dips hard right after the gate turns on at step 2000 (for example, champ_matched goes from 63 modes at 2000 to 54 at 2500). This step is the common failure point.

## Recommendations

- Keep `g_interp_cap` as the sparse-UCD regularizer. Do not port the secant critic as is.
- For a constant-LR champion, try a lower constant LR, or `d_beta2 0.9` alone on the champion. Either would show whether the anneal's stabilising effect can be reached without a schedule.
- If the secant is retried here, measure it against the nearest same-class **mode** (known centers) rather than the nearest batch real, or use t = 0.25 with lam_path 1, so that it stops dominating.
