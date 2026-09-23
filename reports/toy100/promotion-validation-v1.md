# Residual-student promotion validation

The strongest shared transfer recipe passed 18/19 frozen hosts at seed 0;
`residual_student` was its sole failure. Two independent, predeclared
single-field searches produced six configurations that each passed a strict
24-checkpoint replay of `residual_student`. The [promotion protocol](#evidence)
froze their validation rule before any broader runs: verify the saved
residual episode, then run the other nine bottlenecks with the byte-identical
config and frozen source; run a fresh all-19 replay only after those ten
screened hosts pass. A passing residual episode and a separate nine-host
screen do not themselves constitute a full-19 result.

All six variants differ from the 18/19 base by exactly one global field
(besides recipe name). The base has κ=1.0, network floor 0.01, prior floor
0.05, 1600-step network horizon, output noise 0.029 with 20% warmup, and
input noise 0.5 ending at 10% of each host budget. Source commit
`a3be165e2ab47290d35ed98426be77d148f04320`, seed 0, one CPU thread,
host data, architectures, budgets, and thresholds were unchanged.

| Candidate change | Residual suffix | Strict other-nine | Failed other hosts |
| --- | ---: | ---: | --- |
| β₂ 0.999 → 0.9975 | 11 | 6/9 | trajectory, mode hold, stripes |
| Output warmup 0.20 → 0.15 | 20 | 5/9 | trajectory, mode hold, overlap, bars |
| Prior regularization 0.05 → 0.06 | 20 | 5/9 | trajectory, mode hold, unequal mass, stripes |
| Output warmup 0.20 → 0.25 | 18 | 4/9 | trajectory, mode hold, unequal mass, overlap, stripes |
| Output σ 0.029 → 0.0285 | 12 | 6/9 | trajectory, mode hold, overlap |
| Output σ 0.029 → 0.0295 | 18 | 4/9 | trajectory, mode hold, unequal mass, overlap, stripes |

Every residual fix regressed trajectory and mode hold. No variant met the
ten-host promotion rule, so none received a fresh all-19 run or supports a
shared 22-toy pass. This is a bounded result for the declared six variants,
not a claim that no shared recipe exists.

## Evidence

The frozen promotion protocol is local at
`/dev/shm/particlegan-toy-promotion-v1-a3be165/protocol.json` (SHA-256
`b4ff8babcfc10fa017158b440f011b804abfca114f4663f20a8c48acb0f45286`).
Each candidate directory under
`artifacts/toy100-accuracy/compatibility/promotion-v1/` contains its
single-field manifest, exact config, copied residual reference, independent
nine-host episodes, source archives, optimizer/noise receipts, raw log,
strict regrades, and per-file SHA-256 list. RAM originals remain at
`/dev/shm/particlegan-toy-promotion-v1-a3be165/`. All six durable copies
matched their original file digests, and strict regrading after relocation
reproduced both the residual pass and nine-host failure. The
[shared search ledger](shared-recipe-search.md) records all 12 component
replays as subset-only evidence. The `artifacts/` paths are local workspace
evidence, not GitHub links.
