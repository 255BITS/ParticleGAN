# Completed scout comparison

|Model|Min warm Q|Q32|Late Q32|Radial32|Warm passes8/32|Cold late stopped|
|---|---:|---:|---:|---:|---:|---:|
|match_shuffle25_5k|0.011008|0.011099|0.009708|0.912|0/0|0.0%|
|match_shuffle25|0.010901|0.011161|0.010869|0.945|0/0|0.0%|
|uncond_w10|0.009286|0.009391|0.008282|0.759|0/0|0.0%|
|joint_g10|0.008615|0.009129|0.007180|0.770|0/0|0.0%|
|joint_g25|0.008096|0.008096|0.007312|0.922|0/0|0.0%|
|joint_w25|0.004839|0.004889|0.004338|1.405|0/0|0.0%|
|state_w10|0.004429|0.004477|0.004064|0.934|0/0|0.0%|
|joint_w10|0.003995|0.004039|0.003470|2.552|0/0|0.0%|

Extension qualifiers: none.

Full warm passes are out of128 at1024 steps. Q is a continuous diagnostic, not a success probability.

## Held-out history information

|Model|Radius/speed R2 clean|After8 writes|After32 writes|After128 writes|
|---|---:|---:|---:|---:|
|match_shuffle25|0.612/0.942|0.341/0.876|0.127/0.267|-0.007/-0.007|
|match_shuffle25_5k|0.599/0.954|0.343/0.878|0.126/0.436|-0.005/-0.009|
|joint_g10|0.614/0.939|0.370/0.797|0.143/0.350|-0.012/-0.008|
|joint_w10|0.811/0.975|0.453/0.902|0.099/0.210|-0.005/0.002|
|joint_g25|0.614/0.943|0.399/0.860|0.185/0.395|-0.004/-0.011|
|joint_w25|0.799/0.952|0.532/0.839|0.168/0.306|-0.010/-0.009|
|uncond_w10|0.774/0.964|0.445/0.878|0.083/0.230|-0.017/-0.002|
|state_w10|0.640/0.941|0.379/0.840|0.016/0.217|-0.006/0.007|

Probe regression is evaluation-only. Histories are held out; particles come from the learned table. Finite-probe failure is not proof of information erasure.

## One-write read response (prefix32)

|Model|Next-read MSE real/generated|Normalized M gap|K real>fake|Correct-anchor margin > shuffled|K real>wrong-history successor|
|---|---:|---:|---:|---:|---:|
|match_shuffle25|0.00506/0.01496|0.00665|—|—|—|
|match_shuffle25_5k|0.00438/0.01145|0.00635|—|—|—|
|joint_g10|0.00490/0.01355|0.00602|60.3%|57.5%|3.7%|
|joint_w10|0.01093/0.04916|0.01177|49.3%|56.3%|38.7%|
|joint_g25|0.00512/0.01518|0.00645|55.7%|53.0%|6.8%|
|joint_w25|0.01076/0.04581|0.01457|62.0%|54.2%|48.5%|
|uncond_w10|0.00822/0.03529|0.01026|52.1%|0.0%|50.6%|
|state_w10|0.01358/0.06418|0.01789|64.1%|53.6%|36.8%|

MSE is evaluation-only. K margins are not calibrated across separately trained models; hybrid/anchor interventions are descriptive.
