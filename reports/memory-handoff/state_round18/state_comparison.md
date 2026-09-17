# Matched state and clock comparisons

|G update|Q without D clock|Q with D clock|Clock change|
|---|---:|---:|---:|
|embedded|0.005087|0.007584|+49.1%|
|intent|0.000988|0.001050|+6.2%|
|hybrid|0.000973|0.001501|+54.2%|
|Shared D memory baseline|0.010901|0.005849|-46.3%|

Q is minimum warm1024 quality over prefixes8/32, not a probability.

|Internal update vs embedded observation control|Q change|
|---|---:|
|intent8|-80.6%|
|hybrid8|-80.9%|
|intent8_dclock|-86.2%|
|hybrid8_dclock|-80.2%|

## G memory: held-out nonlinear process probes

|Model|Clean radius/speed R2|After8|After32|After128|Real-write128|
|---|---:|---:|---:|---:|---:|
|embedded8|0.441/0.879|0.263/0.519|0.065/0.001|-0.014/-0.007|0.432/0.883|
|embedded8_dclock|0.427/0.875|0.263/0.560|0.031/0.089|-0.003/0.000|0.423/0.873|
|intent8|0.430/0.874|0.264/0.337|0.050/0.052|-0.005/-0.004|0.436/0.874|
|intent8_dclock|0.414/0.862|0.159/0.022|0.006/0.002|-0.000/-0.003|0.422/0.860|
|hybrid8|0.425/0.874|0.248/0.478|-0.025/-0.015|-0.003/-0.004|0.431/0.874|
|hybrid8_dclock|0.418/0.866|0.158/0.253|0.010/0.002|-0.009/-0.006|0.419/0.867|

Independent held-out histories; particles use the learned table. Regression belongs only to diagnostics. Finite probes do not establish information-theoretic erasure.

### Real-trained probe transfer into generated G state

|Model|After1 radius/speed R2|After8|After32|After128|Generated128 with particle|
|---|---:|---:|---:|---:|---:|
|embedded8|0.394/0.855|0.033/0.317|-0.281/-0.359|-0.809/-0.632|-0.010/-0.007|
|embedded8_dclock|0.377/0.850|0.062/0.283|-0.410/-0.280|-0.883/-1.118|-0.009/-0.001|
|intent8|-1.602/0.115|-28.991/-1.782|-71.087/-0.877|-46.723/-1.032|-0.006/-0.002|
|intent8_dclock|-9.113/-0.714|-103.891/-0.669|-140.036/-4.196|-121.792/-1.880|-0.003/-0.003|
|hybrid8|-0.455/0.510|-17.912/-1.432|-46.177/-0.142|-102.518/-1.081|-0.001/-0.001|
|hybrid8_dclock|-1.950/-0.297|-14.120/-2.152|-18.563/-2.789|-16.591/-1.911|-0.008/-0.008|

## D memory: held-out nonlinear process probes

|Model|Clean radius/speed R2|After8|After32|After128|Real-write128|
|---|---:|---:|---:|---:|---:|
|baseline_dclock|0.588/0.951|0.364/0.847|0.105/0.317|-0.012/-0.008|0.585/0.951|
|embedded8|0.793/0.979|0.354/0.822|0.061/0.007|-0.012/-0.018|0.779/0.975|
|embedded8_dclock|0.828/0.980|0.380/0.892|0.049/0.220|-0.003/-0.008|0.820/0.977|
|intent8|0.748/0.972|0.354/0.777|0.050/0.041|-0.009/-0.002|0.728/0.973|
|intent8_dclock|0.828/0.979|0.335/0.832|0.040/0.184|0.004/-0.008|0.813/0.976|
|hybrid8|0.753/0.974|0.336/0.781|-0.009/-0.075|-0.009/-0.003|0.735/0.972|
|hybrid8_dclock|0.826/0.976|0.370/0.835|-0.059/0.037|-0.006/-0.007|0.822/0.975|

Independent held-out histories; particles use the learned table. Regression belongs only to diagnostics. Finite probes do not establish information-theoretic erasure.

## Persistent G-state read interventions (prefix32)

|Model|Normal first32 error|Zero Mg|Shuffle Mg|
|---|---:|---:|---:|
|embedded8|1.379|1.222|1.527|
|embedded8_dclock|1.333|1.264|1.602|
|intent8|2.463|1.223|2.736|
|intent8_dclock|3.000|1.251|3.274|
|hybrid8|2.523|1.203|2.888|
|hybrid8_dclock|2.710|1.231|2.983|

D zero/shuffle metrics are exactly equal to normal256 for all separated scouts. G interventions replace the read state at every generated step; sensitivity is not proof of useful retention.

## Behavioral response to changed real prefixes

|Model|Radius response median (ideal1)|Speed response median (ideal1)|Both directions correct|
|---|---:|---:|---:|
|match_shuffle25|0.02537|0.001291|9.4%|
|match_shuffle25_5k|-0.01336|-0.002073|14.1%|
|baseline_dclock|-4.051e-08|4.186e-09|0.0%|
|embedded8|0|0|0.0%|
|embedded8_dclock|0|0|0.0%|
|intent8|0|0|0.0%|
|intent8_dclock|0|0|0.0%|
|hybrid8|0|0|0.0%|
|hybrid8_dclock|0|0|0.0%|

Responses measured late in1024-step evaluation. Direction here is late mean sign, weaker than full-orbit success.
