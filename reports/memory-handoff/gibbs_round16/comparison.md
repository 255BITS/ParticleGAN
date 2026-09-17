# Completed scout comparison

|Model|Min warm Q|Q32|Late Q32|Radial32|Warm passes8/32|Cold late stopped|
|---|---:|---:|---:|---:|---:|---:|
|match_shuffle25_5k|0.011008|0.011099|0.009708|0.912|0/0|0.0%|
|match_shuffle25|0.010901|0.011161|0.010869|0.945|0/0|0.0%|
|uncond_w10|0.009286|0.009391|0.008282|0.759|0/0|0.0%|
|read_w01|0.009212|0.009212|0.008473|0.801|0/0|0.0%|
|joint_match_g10|0.009033|0.009033|0.008753|0.844|0/0|0.0%|
|joint_g10|0.008615|0.009129|0.007180|0.770|0/0|0.0%|
|state_match_g10|0.008340|0.008340|0.007892|0.909|0/0|0.0%|
|state_g10|0.007432|0.007504|0.006981|1.173|0/0|0.0%|
|read_g10|0.006940|0.006940|0.006194|0.972|0/0|0.0%|
|read_w10|0.004944|0.004944|0.004320|1.410|0/0|0.0%|
|state_w10|0.004429|0.004477|0.004064|0.934|0/0|0.0%|

Extension qualifiers: none.

Full warm passes are out of128 at1024 steps. Q is a continuous diagnostic, not a success probability.

## Held-out history information

|Model|Radius/speed R2 clean|After8 writes|After32 writes|After128 writes|
|---|---:|---:|---:|---:|
|match_shuffle25|0.612/0.942|0.341/0.876|0.127/0.267|-0.007/-0.007|
|match_shuffle25_5k|0.599/0.954|0.343/0.878|0.126/0.436|-0.005/-0.009|
|state_g10|0.617/0.942|0.385/0.820|0.121/0.302|-0.012/0.001|
|joint_match_g10|0.589/0.937|0.379/0.814|0.044/0.089|-0.016/-0.009|
|state_match_g10|0.599/0.940|0.377/0.852|0.129/0.364|-0.008/0.003|
|read_g10|0.600/0.940|0.374/0.817|0.112/0.126|-0.013/0.001|
|read_w01|0.651/0.956|0.382/0.876|0.113/0.252|-0.007/-0.014|
|read_w10|0.891/0.986|0.506/0.939|0.032/0.207|-0.006/-0.002|
|joint_g10|0.614/0.939|0.370/0.797|0.143/0.350|-0.012/-0.008|
|uncond_w10|0.774/0.964|0.445/0.878|0.083/0.230|-0.017/-0.002|
|state_w10|0.640/0.941|0.379/0.840|0.016/0.217|-0.006/0.007|

Probe regression is evaluation-only. Histories are held out; particles come from the learned table. Finite-probe failure is not proof of information erasure.

## One-write read response (prefix32)

|Model|Next-read MSE real/generated|Normalized M gap|K real>fake|Correct-anchor margin > shuffled|K real>wrong-history successor|
|---|---:|---:|---:|---:|---:|
|match_shuffle25|0.00506/0.01496|0.00665|—|—|—|
|match_shuffle25_5k|0.00438/0.01145|0.00635|—|—|—|
|state_g10|0.00503/0.01395|0.00589|58.3%|56.4%|4.1%|
|joint_match_g10|0.00528/0.01641|0.00703|48.9%|49.0%|100.0%|
|state_match_g10|0.00499/0.01466|0.00635|47.3%|49.6%|100.0%|
|read_g10|0.00553/0.01834|0.00720|63.8%|61.3%|11.0%|
|read_w01|0.00501/0.01464|0.00610|65.7%|62.9%|6.0%|
|read_w10|0.01101/0.04694|0.00926|72.3%|68.3%|85.8%|

MSE is evaluation-only. K margins are not calibrated across separately trained models; hybrid/anchor interventions are descriptive.
