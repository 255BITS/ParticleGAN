# Hopfield read over the particle prior

This study tests whether content-addressed retrieval from a learned particle table improves imbalanced mode weights and particle efficiency while retaining within-mode variation. The user chose **one seed per arm**, consistent with `AGENTS.md`; there are no repetitions that differ only by seed. Results are a screen, without training-seed error bars or the originally proposed ≥2-of-3-seeds robustness claim.

The first gate consists of two imbalanced M=100 runs at 7,000 steps: uniform read and Hopfield β=16. Continue only if final `TV_uniform − TV_hopfield ≥ 0.20`. Otherwise report the pair and stop. A passing gate permits the remaining single-seed grid: datasets {uniform, imbalanced}, reads {uniform, Hopfield β=4, 16, 64, learned β initialized at 16}, M {100, 1,000, 20,000}: **30 unique arms including the gate pair**. The discriminator recipe stays fixed: RpGAN logistic, `b_cap`, Fourier-2 D, Adam β1=0, EMA 0.995 and delayed cosine decay.

The target weights are exactly `100 ** (arange(100) / 99)`, normalized and assigned to cells by one seed-0 permutation independent of the training seed. Their smallest weight is **0.04589%**, largest **4.58896%**, and top ten total **37.55557%**. These correct the proposal's approximate 0.1% and 50%. At 100,000 evaluations, the lightest mode has about 45.9 expected draws before the HQ filter. Each run persists the permuted target vector, and measures its TV noise floor with 20 independent draws of 100,000 true-mixture samples. Noise-floor mean ± sd reflects sampling noise, not uncertainty across training seeds.

Uniform retrieval retains the existing sampling path. Hopfield retrieval uses Gaussian queries in the existing latent coordinates, `softmax(exp(log_beta) * q @ Z.T) @ Z`, with no projections or additional generator noise. Both discriminator and generator updates use the selected read. Hopfield VICReg is deliberately applied to the **full particle table**, whereas the uniform path retains its sampled unique-index regularization. Therefore any observed benefit concerns this complete intervention, including that regularization difference.

A learnable β belongs to the prior optimizer, its log is clamped to `[0, log(256)]` after each update, and its EMA is evaluated alongside EMA particles and generator. Evaluation uses 100,000 fresh read samples every 100 steps. The TV and KL histograms count HQ samples only; HQ must be reported alongside these conditional metrics. Read entropy and maximum weight use 4,096 queries, while dead-particle argmax counts use 100,000. Interpolation uses 256 query pairs × 32 points.

`steps_to_tv` is the first evaluation in the final uninterrupted suffix below TV 0.03, rather than the first transient crossing. A missing crossing is reported as greater than the actual run horizon. The quality floor is modes=100, HQ≥0.98 and core σ ratio≥0.80. The interpolation target is HQ≥0.90. At M=100, hard retrieval provides only 100 deterministic generator outputs; excellent HQ can coexist with vanishing within-mode spread.

There are two limits to the hypothesis worth preserving in interpretation. With exactly 100 deterministic uniformly sampled particles and all 100 modes represented, mode weights are forced to be equal. If some modes are dropped, the resulting HQ-conditional histogram may be nonuniform, so TV by itself does not prove healthy reweighting. Also, in the hard-read limit, centered isotropic Gaussian queries partition query space into conical argmax regions: particle selection mass is their Gaussian solid angle. It is not simply Gaussian density at each particle's radius. Changing radii can change those boundaries, and finite-temperature weighted sums further change decoded samples; these distinctions matter when interpreting VICReg or β behavior.

The analyzer is `experiments/analyze_hopfield.py`. It consumes completed `runs/*/summary.json`, per-run `metrics.jsonl` and persisted target weights, and writes `TABLE.md`, `LEADERBOARD.md`, JSON/CSV leaderboards, `REPORT.md`, a machine-readable kill decision, TV and entropy histories, a sorted true-versus-generated mode-weight plot, and scatter GIFs for the gate pair and best measured Hopfield arm. The leaderboard has one row per run, without seed averaging. The best TV row must also satisfy the quality floor before it is a candidate result.

```sh
mkdir -p results/hopfield
.venv/bin/python -u experiments/run_hopfield.py > results/hopfield/PIPELINE.log 2>&1
# In another terminal:
tail -F results/hopfield/PIPELINE.log results/hopfield/runs/*/log.txt
# Regenerate the analysis from completed summaries:
.venv/bin/python experiments/analyze_hopfield.py --runs-dir results/hopfield/runs
```

If the gate passes, compare M=1,000 Hopfield fidelity and quality to uniform M=20,000; if the latter fails sustained TV convergence at 7k, extend that uniform arm to 21k before calling its migration slow. Similar convergence timing would support particle efficiency rather than a unique mode-weight-fidelity advantage. A healthy-read TV plateau can motivate `lambda_ep` {0.1, 1.0}; additional noise inputs, projections, sparse retrieval, and discriminator changes remain out of scope. No follow-up grid or rescue tuning follows a failed gate.

## Completed kill test

Seed 1234, 7,000 steps, imbalanced target, M=100:

| Read | TV ↓ | Modes | HQ | Core σ ratio | Interpolation HQ |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.2121 | 45 | 0.8316 | 0.0580 | N/A |
| Hopfield β=16 | 0.4229 | 38 | 0.6352 | 6.1938 | 0.5908 |

**Stop.** Hopfield worsened TV by 0.2109, instead of improving it by at least 0.20. Neither arm approached the TV<0.03 target or met the quality floor. The measured true-mixture sampling floor was 0.00998 ± 0.00097 across 20 draws of 100,000 samples. The remaining 28 grid arms and longer baseline were not run, as required by the gate.

Hopfield retrieval was close to hard selection: mean maximum weight 0.9407, effective particle count 1.2012, and 70% of particles never selected as argmax among 100,000 queries. This is not the predicted diffuse global-mean failure. Interpolation HQ of 0.5908 also missed 0.90. Uniform sampling produced nearly delta-function within-mode spread. Hopfield's σ ratio of 6.1938 is not evidence of restored Gaussian variation: the estimator is an unweighted average across nearest-center assignments, including low-mass diffuse regions, and HQ was only 0.6352. Inspecting the saved samples found 57 of 85 qualifying assigned modes had core ratios above 2, but these held just 9.67% of generated samples; broad low-mass regions dominate this average.

Retain the existing recipe and stop this proposed study at its predeclared gate. These results reject its M=100/β=16 screening prediction for this seed; they do not establish what other temperatures or larger tables would do. See the [full report and plots](../results/hopfield/REPORT.md), [leaderboard](../results/hopfield/TABLE.md), and the [uniform](../results/hopfield/imbalanced_m100_uniform_s1234.gif) and [Hopfield](../results/hopfield/imbalanced_m100_hopfield_b16_s1234.gif) scatter animations.
