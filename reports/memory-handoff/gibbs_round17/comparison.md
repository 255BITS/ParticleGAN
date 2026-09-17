# Completed scout comparison

|Model|Min warm Q|Q32|Late Q32|Radial32|Warm passes8/32|Cold late stopped|
|---|---:|---:|---:|---:|---:|---:|
|match_shuffle25_5k|0.011008|0.011099|0.009708|0.912|0/0|0.0%|
|match_shuffle25|0.010901|0.011161|0.010869|0.945|0/0|0.0%|
|gibbs1_joint25|0.007228|0.007228|0.005794|0.891|0/0|0.0%|
|gibbs1_arch|0.005940|0.005990|0.005176|1.182|0/0|0.0%|
|gibbs1_joint10|0.005587|0.005755|0.004831|1.201|0/0|0.0%|
|gibbs3_joint25|0.005056|0.005176|0.004630|1.969|0/0|0.0%|
|gibbs3_arch|0.004353|0.004511|0.003871|2.027|0/0|0.0%|
|gibbs3_joint10|0.003542|0.003690|0.003207|2.129|0/0|0.0%|

Extension qualifiers: none.

Full warm passes are out of128 at1024 steps. Q is a continuous diagnostic, not a success probability.

## Held-out history information

|Model|Radius/speed R2 clean|After8 writes|After32 writes|After128 writes|
|---|---:|---:|---:|---:|
|match_shuffle25|0.612/0.942|0.341/0.876|0.127/0.267|-0.007/-0.007|
|match_shuffle25_5k|0.599/0.954|0.343/0.878|0.126/0.436|-0.005/-0.009|
|gibbs1_arch|0.613/0.940|0.343/0.830|0.122/0.299|-0.007/0.000|
|gibbs1_joint10|0.622/0.942|0.379/0.820|0.086/0.268|-0.013/-0.003|
|gibbs1_joint25|0.603/0.940|0.382/0.824|0.108/0.321|-0.007/-0.002|
|gibbs3_arch|0.600/0.939|0.333/0.824|0.050/0.264|-0.010/0.000|
|gibbs3_joint10|0.613/0.941|0.366/0.838|0.079/0.285|-0.005/-0.007|
|gibbs3_joint25|0.607/0.941|0.332/0.845|0.141/0.274|-0.001/-0.004|

Probe regression is evaluation-only. Histories are held out; particles come from the learned table. Finite-probe failure is not proof of information erasure.

## One-write read response (prefix32)

|Model|Next-read MSE real/generated|Normalized M gap|K real>fake|Correct-anchor margin > shuffled|K real>wrong-history successor|
|---|---:|---:|---:|---:|---:|
|match_shuffle25|0.00506/0.01496|0.00665|—|—|—|
|match_shuffle25_5k|0.00438/0.01145|0.00635|—|—|—|
|gibbs1_arch|0.00500/0.01461|0.00616|—|—|—|
|gibbs1_joint10|0.00519/0.01569|0.00687|—|—|—|
|gibbs1_joint25|0.00530/0.01596|0.00667|—|—|—|
|gibbs3_arch|0.00605/0.02125|0.00743|—|—|—|
|gibbs3_joint10|0.00568/0.01923|0.00720|—|—|—|
|gibbs3_joint25|0.00566/0.01971|0.00744|—|—|—|

MSE is evaluation-only. K margins are not calibrated across separately trained models; hybrid/anchor interventions are descriptive.
