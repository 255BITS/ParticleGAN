Gaussian sign pairing removes only a small part of total generator gradient variance in a frozen six-site caption-transfer probe. The predeclared full-generator variance gate failed: ratio .921008 versus the required maximum .75. No antithetic quality training followed.

The portable example uses public `initialize_`, `get_recipe`, `RoutedRows`, `E22Policy`, `routed_generate`, and `load_state_dict`. It requires no local checkpoints or caption fixture. Run the fixed DV12 limitation fixture with:

```bash
PYTHONPATH=. python examples/e22_routed_antithetic_variance.py --fixture routed-dv12 --bf16
```

This command reports ownership checks separately from the variance result. Exit0 means variance PASS; exit1 means a completed variance FAIL. An exception means incomplete execution. Both fixtures are capped at two B4 batches and16 private DV12/Gaussian pairs per batch, sigma .125, and zero training updates. The threshold is the same .75 for both. The first fixed `routed-dv12` measurement returned variance FAIL at .978547 while all ownership checks passed. Its code-to-hidden coefficient1 and streams were declared before execution and were not scanned.

The default `--fixture software` uses a fixed small code-to-hidden coefficient1e-5 to test Gaussian estimator transport and state ownership. Its variance PASS is a software measurement, not a Supra improvement. Both fixtures route two successive sites through one128×4 bank. In the DV12 fixture, the first site's perturbed code changes the second site's host inputs, queries, and full-generator Jacobian. These are live native-controller perturbations, not synthetic gradient noise or a replacement quadratic loss.

Each pair runs the routed host once with a fresh isolated DV12 draw, then averages the unchanged native generator losses at positive and negative Gaussian output noise. The two signs share the same prediction; consecutive pairs draw new DV12. The estimator retains the symmetric Gaussian expected game. It does not change the critic, controller, output guards, objective, or update schedule, and it cannot remove a biased mean critic force.

The primary statistic is the mean squared centered gradient norm over every live generator-role parameter, centered separately within each fixed batch. The variance ratio divides the mean antithetic variance by the mean variance of the32 signed individual gradients. Router, table, F32 residual-upstream, clean-fixed Gaussian variance, and signal energy are reported separately. This is a variance candidacy test, not a convergence criterion.

The captured actual caption task gave:

| Gradient role | Total DV12 + Gaussian ratio | Clean-fixed Gaussian ratio |
| --- | ---: | ---: |
| Full generator | .921008 | .032708 |
| Router | .871926 | .021726 |
| Particle table | .949100 | .026611 |
| Residual upstream | .815832 | .019438 |

Both actual generator batches failed individually (.879850/.941404). Total single generator variance was .156754 versus .014000 for clean-fixed Gaussian variance. Gaussian pairing suppressed97% of the latter but improved total variance by only7.9%. This supports DV12-associated variation in this frozen panel. The variance difference is not an independent component decomposition: DV12 also changes host Jacobians and critic inputs. It does not prove that DV12 causes the convergence gap or support removing it. No per-site code/displacement geometry was measured.

BF16 host transport introduces another distinction: averaging losses before backward can differ from averaging gradients after their individual BF16 transport. The actual probe found1.18–1.70% relative RMS differences in generator gradients, while the F32 residual-upstream identity agreed within1.87e-9. The actual averaged-loss gradient supplies the primary variance statistic.

The example uses `autograd.grad` throughout and verifies unchanged native/model state, stored gradients, module modes, global CPU/CUDA RNG, caller RNG, and native private streams. Diagnostic streams advance privately; saved state is restored in `finally`. Learned tensors, gradients, and Adam moments must be finite. Native monitor validity is delegated to public `load_state_dict`; byte identity preserves its legitimate NaN placeholders.

Run the software contracts with:

```bash
PYTHONPATH=. python -m pytest -q tests/test_e22_routed_antithetic_variance.py
```

Nine CPU cases passed, including public restoration, live DV12, whole-generator transport, the BF16 distinction, a meaningful routed variance FAIL, unchanged state/gradients/RNG, and restoration after a deadline exception. Existing paired-antithetic algebra tests remain unchanged.

The compact [actual-task receipt](e22_routed_caption_antithetic_failure.json) records the single zero-update CUDA probe, source/artifact hashes, baseline point checks, costs, and limits. It completed in8.963 seconds under a120-second cap. All240 clean saved residuals and four native baseline games/context/subject reductions matched before measuring. This point check does not qualify the earlier caption campaign, which failed its existing CPU audit and remains provisional. Machine-specific adapters, absolute-path cards, tensor archives, and raw logs are outside this portable example. No full-Supra convergence or default-policy claim follows.
