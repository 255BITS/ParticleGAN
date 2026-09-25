# Read-only rare-case diagnosis

The archived `shared_c6` LayerNorm/Softplus β4 episode was replayed with an
observer around evaluation sampling and scoring. The observer calls the original
scorer first, then groups the resulting samples by frozen target component and
particle ID. No diagnostic value reaches the loss, optimizer, schedule or action
controller. [Exact replay checks](checks.json) match the archived recipe,
candidate, original and effective specs, discriminator card, actual optimizer
groups, complete live/EMA observations, actions and verdicts. Only timing fields
are excluded, as in the standard replay validator.

| Step | Unique particles by component | Sampled normalized covariance eigenpairs by component | HQ |
| ---: | --- | --- | ---: |
| 1000 | 138 / 79 / 34 / 5 | .090/.376; .139/.606; .103/.266; .082/1.298 | .997 |
| 1050 | 138 / 79 / 34 / 5 | .073/.214; .143/.487; .110/.296; .031/.988 | .997 |
| 1100 | 138 / 79 / 34 / 5 | .098/.924; .334/2.653; .223/.288; .121/1.977 | .841 |
| 1150 | 139 / 78 / 34 / 5 | .158/1.335; .245/1.420; .121/.245; .429/2.153 | .974 |
| 1200 | 139 / 78 / 34 / 5 | .134/.868; .210/.986; .125/.161; .332/1.688 | .990 |

The final minimum variance failure comes from the 13% component, whose 34
distinct particles are narrow in both directions. The 55% component also fails
in one direction. The 2% component has five particles and exceeds the variance
threshold at the final two observations. Thus the named rare-component test is
failing through general within-mode contraction and changing covariance shape,
not disappearance of the 2% component. At step 1100 the 30% component expands
strongly in one direction and HQ falls below its gate; this suggests oscillatory
shape changes but does not identify a unique causal training mechanism.

[All 24 live and EMA diagnostic observations](diagnostics.json) retain per-mode
sample and unique-particle covariance, eigenvalues, support and particle output
coordinates. [Protocol and exact source](protocol.json) · [source archive](source.tar.gz).
This replay is validation evidence and adds no architecture selection point.
