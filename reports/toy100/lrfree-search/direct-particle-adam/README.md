# Direct-particle Adam: a one-host improvement for `st-10`

The new 22-check screen exposed a specific optimizer failure in `two_pole`: its
12 parameters are the samples themselves, and their gradients fall sharply
after the first updates. The recipe-wide AMSGrad flag retains the initial
second-moment maximum, suppressing the subsequent direct-particle response.

`recipe.patch` changes `Recipe.make_generator_optimizer` so the explicitly
registered `direct_particles` group uses ordinary Adam, while all other
generator, latent-table and critic groups retain the recipe's AMSGrad choice.
It adds no host-specific test or numerical threshold. Apply the patch to
`candidates/dv12-st/package/particlegan/recipes.py` in the LR-free harness;
`overrides.json` is the unchanged `st-10` recipe.

| `st-10` custom host | Baseline | Direct-particle Adam |
|---|---:|---:|
| `two_pole`, 80 updates | FAIL 0/24 | **PASS 7/24**, first 60, final streak 7, final mean_abs .7502 |
| Other seven custom hosts | 5/7 | **5/7**; all 24 observation rows exactly equal, excluding seconds |
| Custom total | 5/8 | **6/8** |

The patched package passed the harness's 1,000-step bitwise CPU parity gate
against its own `GANTrainer` for both scalar and sparse-table configurations.
The eight custom hosts were rerun at their frozen budgets with noisy scoring;
their raw `result.json` files are under `results/`. `summary.json` gives the
recipe hashes and seven exact-row comparisons. The direct-particle run was
repeated after the final patch edit and passed again.

The unchanged `st-10` package passed 11/11 quick gates and 0/3 native tasks.
The patch's direct-particle branch is not reached on any of those 14 tasks;
their previous results therefore imply **17/22**, but they were not rerun for
this package hash. The ring and stationary controls also do not call this
branch. This is a narrow custom-host improvement, **not a native 100-Gaussian
solution**. Do not promote it as a default on this result alone.

The latest native follow-up separately grafted paired-only birth/death onto
the sigma-floor candidate. It improved grid100 centre error .303→.260σ,
maximum covariance eigenvalue 2.553→2.012 and mass TV .061→.039 while
preserving precision .9732→.9737, but still failed its .20σ/1.70 limits.
The other three contemporaneous structural lanes failed grid100 too. Their
full artifacts are in `gan-attempts/formulations-20260928T034532Z` outside
this PR worktree. A possible next native test is a *bounded, held-out*
relative-mass-transfer preflight: for noisy component laws K_i and mixture q,
estimate `ell_i = E_{K_i}[log(q/p)]` independently; a child→parent move has
first-order KL change `ell_parent - ell_child`. Require that difference to be
negative on fresh data before moving, then check frozen accuracy and a
matched-law null. Current birth/death uses an absolute-zero test; the
relative test has not been implemented or run, and existing logs cannot
reconstruct the required component scores.
