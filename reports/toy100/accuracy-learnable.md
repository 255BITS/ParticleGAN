# Shared-core learnable output-noise grid trial

The single grid100 trial used [accuracy_learnable.json](../../configs/toy100/accuracy_learnable.json): the shared candidate recipe with only a new name and `output_noise_learnable: true`. The same 7,000-step seed-1234 budget, batch 2,048, 20,000-particle prior, output-noise initialization 0.029, 20% output warmup, and input-noise schedule were retained. The generator gained one trainable scalar in its ordinary G optimizer group. The complete run evidence (local evidence: `artifacts/toy100-accuracy/learnable-shared/grid/grid100`) includes events, scored samples at every check, and a separate 100,000-draw holdout. Native source hashes in the run's provenance match the source after completion.

The trial **fails** both the original coverage gate and the strict accuracy gate. Neither live nor EMA achieved a single passing evaluation. At 7,000 steps, live has 4/100 modes, HQ 0.1037, and mass TV 0.6096; EMA has 7/100 modes, HQ 0.1859, and mass TV 0.6089. The independent live holdout has mass TV 0.6107 and precision 0.1048. Conditional fidelity measures are undefined because coverage is insufficient.

| Step | Live effective σ | Live modes | Live HQ | Live mass TV | EMA modes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0.9698 | 0 |
| 500 | 0.009382 | 0 | 0.0214 | 0.3171 | 0 |
| 1,000 | 0.020434 | 1 | 0.0249 | 0.7828 | 2 |
| 1,250 | 0.025035 | 1 | 0.0253 | 0.7154 | 0 |
| 1,500 | 0.024275 | 0 | 0.0258 | 0.6886 | 1 |
| 2,500 | 0.015293 | 1 | 0.0278 | 0.5978 | 1 |
| 4,000 | 0.008869 | 1 | 0.0387 | 0.5567 | 0 |
| 5,000 | 0.006820 | 2 | 0.0435 | 0.5012 | 2 |
| 6,000 | 0.005571 | 2 | 0.0439 | 0.5526 | 2 |
| 6,250 | 0.005214 | 3 | 0.0983 | 0.6091 | 5 |
| 6,500 | 0.004834 | 7 | 0.1285 | 0.6056 | 7 |
| 6,750 | 0.004523 | 5 | 0.1250 | 0.6141 | 5 |
| 7,000 | 0.004309 | 4 | 0.1037 | 0.6096 | 7 |

All five terminal live checks (6,000–7,000) fail coverage and accuracy. The learned base σ rises toward the original 0.029 initialization during warmup but then declines to 0.004309 (15% of initialization), while the generator remains collapsed. This shows that letting the original small output noise adapt does not repair the shared-core grid run. It does not isolate a larger noise initialization, removal of warmup, or removal of input noise; those would be a jointly changed mechanism requiring the same transfer and strict-gate checks on all tasks. In the events, each `train` row's noise fields describe the **post-update evaluation amplitude**; the just-completed update used the previous completed-step warmup multiplier. Each `eval` row's `output_sigma` is the amplitude applied to that evaluation draw.
