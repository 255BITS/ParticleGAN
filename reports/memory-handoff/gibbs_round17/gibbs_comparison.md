# Same-event diagnostics

1024 fresh histories, prefix32; all errors below are evaluation-only MSE.

|Model|Point error|Zero M error|Shuffled M error|Zero h output change|Shuffled h output change|K real>fake|K pair>wrong history|
|---|---:|---:|---:|---:|---:|---:|---:|
|match_shuffle25|0.00467|0.75935|1.43632|—|—|—|—|
|match_shuffle25_5k|0.00403|0.75047|1.43848|—|—|—|—|
|gibbs1_arch|0.00456|0.76842|1.45038|0.002983|0.000941|—|—|
|gibbs1_joint10|0.00473|0.73812|1.44949|0.005859|0.000306|56.9%|42.1%|
|gibbs1_joint25|0.00473|0.74787|1.45356|0.005783|0.000091|56.4%|43.9%|
|gibbs3_arch|0.00550|0.72710|1.44893|0.036036|0.022756|—|—|
|gibbs3_joint10|0.00521|0.72885|1.44362|0.035903|0.024996|50.5%|24.9%|
|gibbs3_joint25|0.00544|0.72866|1.42215|0.041547|0.025979|43.3%|22.9%|

## Inference refinement, context held fixed

|Model|1 decode error|2 decodes|3 decodes|7 decodes|Producer vs reencoded h error|
|---|---:|---:|---:|---:|---:|
|gibbs1_arch|0.00456|0.00742|0.00740|0.00740|0.217476|
|gibbs1_joint10|0.00473|0.00483|0.00482|0.00482|0.007212|
|gibbs1_joint25|0.00473|0.00503|0.00503|0.00503|0.010901|
|gibbs3_arch|0.20349|0.00713|0.00550|0.00551|0.000008|
|gibbs3_joint10|0.17351|0.00555|0.00521|0.00521|0.000000|
|gibbs3_joint25|0.15955|0.00551|0.00544|0.00543|0.000001|

Latent distances are representation-specific. Interventions measure local sensitivity and error, not autonomous survival. Extra inference iterations can be out of training support. A good joint rank or small latent discrepancy is not proof of equilibrium or circle completion.
