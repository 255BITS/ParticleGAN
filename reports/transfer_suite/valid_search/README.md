# Valid behavioral toys: search for b_cap3 defaults

**The original b_cap3 recipe now passes 9/9 required and 9/10 practical toys using supported discriminator architectures.** The rare 2% mode is the remaining failure. Different toys may use different D architectures; a single D does not pass them all. Adam beta2=.999 reaches 8/10 with different supported image architectures.

All rows use Rp logistic, b_cap3/κ1.25, prior regularization .05 and no particle L2. Optimizer choices are separate trials under that formulation. Architecture variants stay within a recipe; failures and every untested case remain visible.

| Training recipe | Required live | Data | Images with supported architecture | Practical | Qualified on required |
| --- | ---: | ---: | ---: | ---: | --- |
| Original recipe + supported D architectures | 9/9 | 5/6 | 4/4 | 9/10 | Yes |
| Adam beta2=.999 | 9/9 | 4/6 | 4/4 | 8/10 | Yes |
| Coordinated LR recipe | 7/9 | 4/6 | 2/4 | 6/10 | No |

The main comparison stays at the original update budgets: Gaussian data 1,200, spiral 1,600, images 600 and each required host’s existing budget. [Longer training](../formulations/LONG_TRAINING.md) is a separate toy. Live thresholds and the final five of 24 rule are unchanged. EMA is retained separately.

## What improved

- **Discriminator architecture alone fixes overlap.** D128×3 with Fourier 3 or 4 passes broad, anisotropic, overlap and spiral. G remains 4,610 parameters; D grows from 4,929 to 35,073/35,585. The required and image results reuse unchanged settings and exact archived evidence. This is an architecture improvement under the same formulation, with no extra updates.
- **Smooth discriminator activation fixes unequal width.** Replacing LeakyReLU with Softplus(beta5) in the original D64×2/Fourier2 holds its 4,929 parameters and every training setting fixed. It passes the final seven checks; beta10 also passes. Their full six-data profiles pass 3/6; the formulation uses appropriate D architectures for the other toys. [Architecture results](smooth_discriminator/README.md) · [Reusable critic and reproduction](../../../benchmarks/transfer_suite/smooth_critic_research.md).
- **Adam beta2=.999 is another recipe under the same formulation.** It passes all nine required hosts and fixes overlap with the smaller original vector D. Residual16 fails blobs (HQ 84.4%); transpose12 passes that toy under the same recipe. All four image architecture profiles are retained.
- **The coordinated recipe trades away other passes.** The coordinated recipe solves rare mass and overlap at 256 particles and the original budget, but loses anisotropic data, required trajectory identity and the required eight-mode ring. Its residual16 images pass 2/4.

Across the same 17 passing behavioral cases, mean confirmed-step / budget, choosing the earliest supported architecture per case: Original recipe + supported D architectures=0.594, Adam beta2=.999=0.621. These are inspected development results. More D capacity and concurrent CPU load prevent a wall-time speed claim.

## Evidence and reproduction

[All D architectures](discriminator/README.md) · [18 coordinated recipes](recipes/README.md) · [D/recipe combinations](discriminator_combinations/README.md) · [Adam999 required/image checks](adam999_hosts/README.md) · [Adam999 image architectures](adam999_images/README.md) · [Coordinated recipe across hosts](coordinated_hosts/README.md) · [Resource searches](resources/README.md) · [Softplus refinement](softplus_refinement/README.md) · [512-particle smooth-D checks](smooth512/README.md).

The research host adapter applies the same LR factors to G, D and ParticlePrior parameter groups across hosts, including direct particle-only optimizers. It reproduces the native rare-vector result and the neutral image control exactly before cross-host evaluation; [parity evidence](coordinated_hosts/parity.json.gz) and exact driver source are archived. No production API or defaults changed.

Independent review caught an explicit AE prior-group beta1=.5 overriding the declared Adam pair. Both affected AE cases were rerun with Adam(0,.999) enforced on every parameter group; both pass. The comparison uses only the corrected AE results. Original runs remain as superseded evidence. [Adam999 correction](adam999_group_fix/README.md) · [Coordinated correction](coordinated_group_fix/README.md).

Seed 0 only; no seed sweeps. Every attempted result, source archive, configuration and failure is retained. [Machine-readable comparison](leaderboard.json) · [Artifact validation](validation.json) · [29 focused tests](tests.log).

```bash
python -m reports.transfer_suite.formulations.build
python -m reports.transfer_suite.valid_search.build
```
