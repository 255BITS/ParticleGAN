# Shared κ bracket with the residual-student bottleneck

The preceding network-floor 0.01 recipe passed its nine-host screen but
failed `residual_student` in the full 19-host replay. Four globally shared
critic-cap thresholds, κ=1.10, 1.15, 1.20, and 1.25, were declared before
training. All other fields stayed identical to that 18/19 recipe, including
network LR floor 0.01, prior LR floor 0.05, the 1600-step network horizon,
β₂=0.999, and the same noise schedule. The source was commit `a3be165`,
seed was 0, each host kept its frozen budget, data, architecture, and gates,
and every run used one CPU thread. The predeclared manifest SHA-256 is
`e6ee41ed907094c52528c494a0343f9264c766e554af3e1f5ac822404b5e7c59`.

The expanded screen contains the previous nine bottlenecks plus
`residual_student`. It requires **10/10** before a full 19-host replay.

| Global κ | Strict result | Failed frozen hosts |
| ---: | ---: | --- |
| 1.10 | 6/10 | trajectory, mode hold, overlap, stripes |
| 1.15 | 5/10 | trajectory, unequal mass, overlap, stripes, bars |
| 1.20 | 7/10 | mode hold, unequal mass, overlap |
| 1.25 | 7/10 | mode hold, stripes, blobs |

`residual_student` passes all four rows, but each loses other required
behavior. None qualified for all 19, and none supports a shared 22-toy
claim. The result shows that changing this one global cap threshold does not
reconcile the full19 miss with the previous nine-host passes at the tested
values. It does not rule out other recipe changes.

All 40 episodes passed source, configuration, noise, optimizer-action, and
original-spec integrity regrading. Raw local evidence and the exact failure
curves are retained at
`artifacts/toy100-accuracy/compatibility/network-floor-kappa-bracket-v1/`
and in RAM at `/dev/shm/particlegan-toy-kappa-bracket-v1-a3be165/`. The
durable copy's 81 original files (5,004,237 bytes) match their SHA-256
values; strict regrading after relocation reproduced 6/10, 5/10, 7/10,
and 7/10. The file-hash manifest SHA-256 is
`7e88688097adc737831e1a1f34f098e807079c891422b685d571051146f2bfc3`.
The `artifacts/` paths are local workspace evidence, not GitHub links.
