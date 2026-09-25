# Broad learnable output-noise grid probe

This one bounded joint-mechanism probe used [accuracy_learnable_broad.json](../../configs/toy100/accuracy_learnable_broad.json). Relative to the [first learnable trial](accuracy-learnable.md), only the descriptive name and three noise fields changed: learned output-noise initialization 0.029 → 0.2, output warmup 20% → 0%, and input noise 0.5 → 0. The shared optimizer, architecture, batch 2,048, prior 20,000, seed 1234, and 7,000-step budget stayed fixed. These three noise changes cannot be attributed separately. The complete run (local evidence: `artifacts/toy100-accuracy/learnable-broad/grid/grid100`) includes final-five 20,000-draw checks and a separate 100,000-draw holdout. All ten recorded native source digests still matched after completion.

The original and strict accuracy gates both **fail**, with 0/5 terminal live checks passing. Final live coverage is 14/100 modes, HQ 0.3701, mass TV 0.7188; EMA is 14/100, HQ 0.3846, mass TV 0.7195. The independent live holdout has precision 0.3658 and mass TV 0.7170. Conditional fidelity statistics are undefined because many modes are missing.

| Step | Live effective σ | Live modes | Live HQ | Live mass TV | EMA modes | EMA mass TV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.200000 | 0 | 0.0027 | 0.9600 | 0 | 0.9600 |
| 500 | 0.167938 | 0 | 0.0220 | 0.4220 | 0 | 0.6013 |
| 750 | 0.164637 | 0 | 0.0249 | 0.2567 | 0 | 0.4589 |
| 1,250 | 0.151796 | 0 | 0.0233 | 0.7789 | 0 | 0.3918 |
| 2,500 | 0.033621 | 2 | 0.0779 | 0.4297 | 1 | 0.4418 |
| 4,000 | 0.013610 | 3 | 0.0750 | 0.3839 | 5 | 0.3666 |
| 5,000 | 0.007530 | 12 | 0.1996 | 0.3282 | 22 | 0.3271 |
| 5,500 | 0.005521 | 23 | 0.3555 | 0.3537 | 34 | 0.3514 |
| 5,750 | 0.005194 | 1 | 0.0166 | 0.7089 | 0 | 0.4877 |
| 6,000 | 0.005049 | 7 | 0.1283 | 0.7329 | 1 | 0.6661 |
| 6,250 | 0.004872 | 6 | 0.0961 | 0.7195 | 2 | 0.7198 |
| 6,500 | 0.004712 | 13 | 0.3839 | 0.7213 | 11 | 0.7221 |
| 6,750 | 0.004580 | 13 | 0.3784 | 0.7207 | 12 | 0.7215 |
| 7,000 | 0.004475 | 14 | 0.3701 | 0.7188 | 14 | 0.7195 |

The zero early mode hits alone do not prove collapse: with σ≈0.16, few samples can fall in each small 3σ target disk. Nearest-center mass TV reached 0.2567 at step 750, and the live sample cloud spanned much of the grid. Allocation and tight-mode quality later fluctuated sharply. The learned noise shrank to 0.004475, or 2.2% of its initialization, without sustained fidelity. This joint mechanism therefore cannot be promoted as a common 22-task recipe from the grid evidence. As in the native runner, each `train` row's scale is the post-update evaluation amplitude; the corresponding update used the previous completed-step amplitude.
