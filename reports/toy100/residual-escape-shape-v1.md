# Residual escape shape screen

Six rows were declared before training against the frozen `.01` network-floor, `κ=1` recipe at source commit `a3be165`. Each row changed exactly one numerical field plus its display name. The base config SHA-256 is `52f2971ac493d5d10f0c54666efcb95b8bd31c54544e16cd14f9822f7694e21f`. Each row used the original unpinned local AVX2 environment, seed 0, and the frozen 400-step `residual_student` task. No native 100-mode training was part of this screen.

| Single change | Final identity MSE | Success / wrong pad | Passing suffix | Frozen gate |
| --- | ---: | ---: | ---: | --- |
| Adam β₂ `.995` | `.051443` | `.500 / .500` | 0 | FAIL |
| Adam β₂ `.9975` | `.000902` | `1.000 / 0` | 11 | PASS |
| Prior regularization `.04` | `.047974` | `.583 / .417` | 0 | FAIL |
| Prior regularization `.06` | `.000920` | `1.000 / 0` | 20 | PASS |
| Output noise σ `.0285` | `.000900` | `1.000 / 0` | 12 | PASS |
| Output noise σ `.0295` | `.000951` | `1.000 / 0` | 18 | PASS |

The four passing config paths and exact hashes are in `artifacts/toy100-accuracy/residual-escape-shape-v1-a3be165/predeclared.json`. They were sent to the separate transfer-validation lane as soon as their residual verdicts were available. A residual pass alone does not establish 19/19 or 22/22 compatibility.

For each row, an independent call to the frozen `test_verdict` recomputed the saved verdict exactly from all 24 observations. Configs differ from the base only in their declared field and name; all 118 source digests match the frozen baseline, and every source archive member matches its recorded digest. The artifact directory retains all six configs, complete episode receipts and logs, the original residual baseline, `regrade.json`, and `hashes.json` with verified SHA-256 hashes for all 59 retained files. Each one-host episode summary is `INCOMPLETE` by design, while its individual frozen host verdict is the result shown above.
