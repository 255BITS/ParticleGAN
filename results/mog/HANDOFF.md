# MoG particle-prior study — compaction handoff

## Latest continuation: component-count and budget experiments COMPLETE

This section supersedes the older checkpoint below. The user authorized more
MoG experiments, larger component tables, and longer training to match/beat C0.
The latest AGENTS instruction says no seed-only experiments. **All 21 new runs
used seed 1; do not infer a need to repeat seeds from the historical study.**
No training is running. Current branch remains `feature/mog-particle-prior`.

Read `results/mog/COMPONENT_SCALE.md` and its CSV/plots for the complete results.
17 screened 7k configurations plus four 28k runs; five historical references reused.
All 2,310 new JSONL rows/configs/certificates/digests validated. The baseline
`results/mog/results.csv` and all training code remain unchanged. Four positive-noise
MoGs pass the frozen C0 envelope; none dominates original seed-1 C0 in every measured
quality metric. No pass-rate claims: this is a single-seed configuration screen.

| Run | HQ/real | Width/real | KL | Pass |
|---|---:|---:|---:|---|
| Original 20k C0, 7k, seed 1 | 1.001274 | .7909 | .03750 | yes |
| 20k MoG r=1/40, unstandardized, 7k | 1.000066 | .8258 | .03570 | yes |
| 20k C0, 28k | .999110 | .9438 | .02655 | yes |
| 20k MoG r=1/16, unstandardized, 28k | 1.000046 | .9543 | .02657 | yes |
| 20k MoG r=1/40, unstandardized, 28k | .999778 | .9567 | .02742 | yes |
| 400 MoG r=1/40, standardized, 28k | .999535 | .9264 | .02888 | yes |

The compact MoG uses 50x fewer components and 4x the original training steps;
raw scale drift is 4.11x (flagged), with standardized read scale controlled.
No matched 400-atom 28k control has been run. The 20k r=1/16 at 28k also clears
all historical C0-mean thresholds (.99978 / .84343 / .03478); it is the only new
MoG to do so. Matched 28k atoms are nearly as good: MoG slightly improves HQ and
width, KL differs by only +.000012. Unstandardized 20k raw scale grows only
11–13%; effective sigma/spacing increases, so no inflation escape here.

At 7k, fast optimizer LR multiplier 10/beta .5 fails as N increases. Shipped
LR multiplier 1/beta 0 restores width for N=1,600/6,400, but balance still fails.
More components do not produce a monotonic gain. Both priors remain useful.

User asked what unstandardized means; answered: learned means are read directly,
without per-read centering/std normalization, as in C0, then fixed noise is added.

New scripts: `gen_mog_scale_configs.py` (screen/optimizer/longer stages) and
`analyze_mog_scale.py --final`. Configs: `configs/mog/component_scale/` (17) and
`configs/mog/scale_longer/` (4). Raw outputs in same-named results folders are
gitignored but preserved. Logs: `component_scale.runner.log`,
`component_scale.optimizer.runner.log`, `scale_longer.runner.log`.
Timing: 77s timed run; 4.1min remaining screen; 1.3min optimizer check; 5.2min longer.
Longer runs start fresh and scale cosine to 28k; exact prefixes match parents
(42 logs for each 20k run, 84 for N=400). See `component_scale_prefix_check.json`.

Completed the previously pending NOISE_CHECK narrative and FINDINGS follow-ups.
Recommendation for a future user-authorized continuation: matched 400-atom 28k
control, then improve the compact MoG schedule to reduce training cost; retain
20k MoG and atoms at matched budget as quality references. No more runs launched.

## Historical checkpoint (before this continuation)

## User intent and authority

User requested the fixed-sigma MoG study on a feature branch, with a Stage 0 gate.
Stage 0 and Stage 1 are complete. User then approved smaller-noise experiments and
explicitly said: **“you can run experiments. we may consider a longer training run
too for the most promising. it's trying to learn a lot more with mog.”** Latest
request is to prepare for compaction, then continue experiments.

Do not launch anything new just to fill time. The refinement finished while the handoff was being written. Review the completed
results and polish the final follow-up findings before choosing another experiment. There is authorization for useful further experiments, but no obligation to
run the entire original 120-run Stage 2 blindly. Keep the fixed-sigma design:
no learned/per-component sigma, weights, resampling/birth-death, Hopfield, or real
dataset. Root AGENTS.md: be token efficient, easy-to-tail logs, summarize with
explanations/leaderboard/recommendations; no seed-only searching. The explicitly
approved study uses seeds 1,2,3 per experimental cell, not best-seed selection.

## Workspace and git

- Repo: `/home/martyn/dev/ParticleGAN`
- Branch: `feature/mog-particle-prior`
- Last training-launch commit: `d5e6a40` (r=1/40 refinement configs).
  The subsequent checkpoint commit saves this handoff and current analysis artifacts.
- Prior commits: `d02489e` (smaller-noise analysis + 14k config), `8a03ae0`
  (small-noise configs), `feeecc7` (Stage 1 report), `33b3144` (frozen pass rule).
- Python: `.venv/bin/python`; two NVIDIA RTX A6000s.
- Existing unrelated untracked items must be preserved: `.claude/`,
  `results/hopfield/`, `results/hopfield_recall/`, `results/motion/`, `runs/`,
  `sparse-ucd.log`.
- A detached pre-change reference worktree remains at
  `/tmp/particlegan-mog-reference` (af1843a). Its completed output artifacts are
  in the main repo's `results/mog/stage0/pre_s{1,2,3}/`.

## Active work — FIRST CHECK THIS

All three **N=400, r=1/40, 14k-step** refinement runs are now COMPLETE,
certified, and analyzed. **No training is running.** The runner finished 3/3,
zero execution failures, in 2.5 minutes. All three fail the frozen quality rule.
Mean HQ/real **0.99641**, width/real **0.926**, KL **0.03588**: much closer to C0.
All three refinement seeds meet coverage, width, and KL; **HQ is the only failed
condition**. Per-seed HQ/real: .994721 / .996794 / .997710 vs required .998832.
Per-seed width/real: .931000 / .936349 / .911642. KL: .035759 / .036350 / .035540.
A plausible next bounded experiment is a slightly smaller fixed radius (e.g.
r=1/44 at 14k), using the width headroom to close the remaining HQ gap; alternatively
more budget at r=1/40. Neither has been launched or promised. Preserve the fixed
criterion and use all three seeds when comparing cells.
All 15 new runs and 1470 trace rows have been validated. Analysis CSV has 18 rows
including three reused Stage 1 references. Historical exec session was 50307.
Use logs and certified completion files rather than assuming they are done:

```bash
tail -F results/mog/refine_noise.runner.log
tail -F results/mog/refine_noise/n400_r1over40_14k_s1/log.txt
ps -eo pid,etime,args | rg 'experiments/(run_grid|train_100gaussians)\.py'
```

Launch command already issued (do not duplicate):

```bash
.venv/bin/python -u experiments/run_grid.py \
  --configs 'configs/mog/refine_noise/*.yaml' \
  --trainer experiments/train_100gaussians.py \
  --gpus 0,1 --workers_per_gpu 2 > results/mog/refine_noise.runner.log 2>&1
```

This is one targeted adaptive refinement after the 14k r=1/32 result: reduce
remaining spread slightly. Its justification was announced before launch. It is
not an originally preregistered grid point. All three seeds retain LR multiplier
10 (actual initial particle LR .06), particle beta1=.5, standardized reads.

## Frozen pass criteria

`configs/mog/stage1_criteria.json` freezes thresholds and the Stage 0 CSV hash.
User changed acceptance to **original or better**, interpreted as the observed
three-seed C0 envelope so all three original reference runs pass:

- modes = 100
- HQ / HQ_real >= **0.9988322835680561**
- width / width_real in **[0.7909158604586308, 1.2090841395413692]**
- HQ-only KL <= **0.03750414025263102**

`passed` / `passed_baseline` use this rule. `passed_strict` retains the original
design (.98 HQ ratio, .9–1.1 width, .01 KL). `passed_c0_mean` is analysis-only
comparison to C0 means. Thresholds must not be moved after observing results.
Keep Stage 0 `results/mog/results.csv` unchanged: it is the frozen baseline input.

## Completed results

Every number below is a cell mean over three seeds. Width and HQ columns are
model/real ratios. All use final 200k EMA samples, noise enabled when sigma>0.

| N | nominal r | steps | HQ/real | width/real | KL | pass |
|---|---|---:|---:|---:|---:|---|
| 20000 C0 | 0, no standardization | 7000 | .99978 | .84343 | .03478 | 3/3 under new rule |
| fresh Gaussian C1 | n/a | 7000 | .06069 | 10.189 | .39484 | 0/3 |
| 100, Stage 1 winner | 1/8 | 7000 | .29137 | 4.400 | .37679 | 0/3 |
| 400, Stage 1 winner | 1/8 | 7000 | .38016 | 3.373 | .15626 | 0/3 |
| 400 atoms control | 0 | 7000 | .99171 | .393 | .02754 | 0/3 |
| 400 | 1/32 | 7000 | .96220 | 1.154 | .04966 | 0/3 |
| 400 | 1/16 | 7000 | .65407 | 2.113 | .08248 | 0/3 |
| 400 | 1/32 | 14000 | .97851 | 1.066 | .03591 | 0/3 |
| 400 | 1/40 | 14000 | .99641 | .926 | .03588 | 0/3 |

- C0 regression: all 70 original log entries and final EMA G/prior tensors are
  bit-identical to pre-change for all 3 seeds. Sigma=0, standardize=False.
- Exact all-atom C0 KL: .036844 / .033613 / .033031. Original mass imbalance is
  real, not a regression. Historical `hist_kl` includes all samples; the new KL
  only includes HQ samples. Saved pre-change outputs were measured both ways.
- Stage 1: 30 LR runs, six new beta=.5 runs, six beta=0 comparisons reused.
  36 unique training runs, 0/36 pass; all fail HQ, width and balance individually.
  Best LR=10x at both N. Best beta1=0 at N=100, .5 at N=400 (small preference).
- All new N=400 noise/longer runs keep that selected optimizer. The N=400 atoms
  control is NOT C0: standardized reads and this aggressive optimizer remain on.
- Low noise is dramatically better. 14k improves width and balance further, but
  r=1/32 still has too much non-HQ mass. r=1/40 at 14k gets much closer, but still 0/3 under the frozen rule.
- 14k runs restart from identical seeds and scale the delayed cosine schedule:
  anneal starts at 8400 vs 4200. They are NOT continuation from EMA checkpoints.
  The first 42 original log entries (through step 4100) are identical in all
  three matched 7k/14k r=1/32 pairs: `longer_prefix_check.json`.
- Timing: small run ~68–69 sec solo; 4 concurrent ~77 sec each. Noise screen
  batch 2.6 min after first timed run. Three 14k runs took 2.4 min. Refinement
  should take roughly the same time.

## User's conceptual question / overlap hypothesis

Explained: each sample chooses a Gaussian component, draws one latent point,
then G sees that point without component identity. Mixture means selecting a
component, not averaging draws. Gaussian clouds overlap; not literally separate
manifolds. User suspects non-overlap is desirable.

Current evidence: distinguish **same-output-mode overlap** from different-mode
ambiguity. Many learned component means clump within a mode. At Stage 1 N=400,
r_eff was ~4.63 despite nominal .125, with ~91% of evaluable nearest neighbors
having the same majority output mode. Do not confuse this with the r=2 Gaussian
control. Raw std drift >2x flags also matter (selected beta=.5 has ~3.26x).

`experiments/audit_mog_overlap.py` computes the exact shared-Gaussian posterior
for 20k CPU latent draws per run, then estimates E[1-max posterior] for component
identity and for components grouped by their observed HQ-majority output mode.
This is a post-training diagnostic, not a new training loss or a certified bound.
Unobserved-HQ components use an explicit unknown label. Figures so far:

| r / steps | component ambiguity | output-mode ambiguity | generated non-HQ |
|---|---:|---:|---:|
| 1/8, 7k | .52268 | .00019877 (~.02%) | .62405 |
| 1/16, 7k | .27380 | ~2.3e-11 in sampled diagnostic | .35318 |
| 1/32, 7k | .12982 | negligible in sampled diagnostic | .04847 |
| 1/32, 14k | .19911 | negligible in sampled diagnostic | .03234 |

Thus different-destination overlap does not appear to explain most non-HQ mass
in these learned models. G has trouble shaping/contracting otherwise identifiable
clouds. Near-zero Monte Carlo estimates are not proofs of globally zero overlap.
At 14k r=1/32, ~99.9% of component centers are HQ, vs noisy HQ .96766.
The `bridge=1-hq` metric also includes over-wide within-mode tails, not just walls
between modes. Avoid claiming all such mass is tearing/overlap.

## Code and semantics

- `particlegan/particle_prior.py`: MoGParticlePrior, fixed sigma/d0 buffers,
  differentiable standardized `means()`, inherited index RNG semantics, no noise
  RNG at r=0. `lib/particle_prior.py` is a compatibility reexport.
- `examples/100gaussians.py`: shared training loop, full raw-table VICReg for
  N<=1024, independent particle LR/beta, seeded fixed scatter eps, JSONL metrics.
- `experiments/train_100gaussians.py`: existing config wrapper, final suite,
  200k final samples, checkpoints, source archives, provenance.
- `lib/mog_metrics.py`: metrics, geometry, per-component audit, frozen pass rules.
- `experiments/gen_mog_configs.py`: stages stage0, stage1_lr, stage1_beta,
  noise_check, longer, refine_noise.
- `experiments/analyze_mog_stage1.py`: committed Stage 1 analysis; helper
  audit_component_centers now accepts output_path to avoid overwriting Stage 1.
- `experiments/analyze_mog_noise.py`: newer analyzer, updated
  to include refinement and require all three refinement completions for final.
- `experiments/audit_mog_overlap.py`: new overlap-diagnostic script, saved with this checkpoint.
- Width reuses `lib.toy_metrics.per_mode_core_ratio`: nearest-mode assignment,
  coordinate-wise median center, median radius / sqrt(2 ln2) / .03, equal average
  over modes with >=50 samples. No HQ truncation. Main modes >=10 HQ samples.
- For the r=0 N=400 exact all-table diagnostic, its n=400 sample count makes
  >=10 coverage / >=50 width unsuitable. Judge this control on primary 200k
  metrics, not `deterministic_modes` or `deterministic_width`.
- Raw VICReg remains scale-dependent; standardization does not remove it.
- The bridge expression is labeled a heuristic, not a valid universal bound.

## Exact continuation steps

1. Training is complete. Read the final `noise_check_results.csv`, `NOISE_CHECK.md`,
   and `noise_check_overlap.csv`. All 15 new runs and 1470 trace rows were validated;
   do not rerun completed training just to recover state.
2. The following commands were already run successfully after refinement. Rerun
   them only after changing analysis or when additional runs justify it:

```bash
.venv/bin/python experiments/analyze_mog_noise.py --phase final \
  > results/mog/noise_check.analysis.log 2>&1
.venv/bin/python experiments/audit_mog_overlap.py \
  > results/mog/noise_check_overlap.log 2>&1
```

3. Follow-up reports/plots now include the refinement. Their result tables are
   current, but narrative completion described below remains outstanding.
4. Extend NOISE_CHECK report generator to incorporate overlap findings,
   refinement outcome, concrete recommendation, and prefix validation as needed.
   It currently has the core tables/diagnostics/deviations but no final overlap
   section or outcome-specific recommendation. Preserve it as reproducible code.
5. Append completed follow-up findings to FINDINGS.md (currently ends at Stage 1).
   Clearly distinguish 7k vs 14k, and the adaptive r=1/40 addition. No claim that
   14k results match C0 at equal training budget.
6. Raw artifact folders have been added to results/mog/.gitignore and the
   current scripts, reports, CSV/JSON/plots and handoff checkpointed. Commit any
   subsequent narrative/analysis changes; do not stage unrelated user files.
7. Give a concise updated leaderboard/explanation/recommendation. Based on the
   final data, choose the next scientifically useful experiment if continuing;
   don't silently loosen acceptance thresholds or launch the full grid on the
   assumption of success. The user has allowed further experiments.

## Existing artifacts

- Stage 0: `results/mog/STAGE0.md`, `results.csv`, regression.json, original_kl.json.
- Stage 1: `STAGE1.md`, `STAGE1_RUNBOOK.md`, stage1_results.csv,
  stage1_leaderboard.csv, stage1_winners.json, optimizer/width-HQ plots.
- Follow-up (checkpointed; narrative additions pending): `NOISE_CHECK.md`, noise_check_*.csv/json/png,
  longer_prefix_check.json. Overlap log is ignored, CSV should be tracked.
- Test status before follow-ups: **35 passed + 13 subtests**. Follow-ups changed
  configs/analysis only, not training code. Exact generated-config differences
  were verified: sigma/out_dir for noise checks; epochs/out_dir for 14k;
  sigma/out_dir for refinement.
