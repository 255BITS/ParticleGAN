# Two-round Gibbs-inspired leaderboard

12 fresh2k scouts, saved2k and5k references. All cold/warm full passes0/128. No qualifying extensions.

|Model|Steps|Min warm Q|Q32|LateQ32|Radial32|
|---|---:|---:|---:|---:|---:|
|match_shuffle25_5k|5000|0.011008|0.011099|0.009708|0.912|
|match_shuffle25|2000|0.010901|0.011161|0.010869|0.945|
|uncond_w10|2000|0.009286|0.009391|0.008282|0.759|
|read_w01|2000|0.009212|0.009212|0.008473|0.801|
|joint_match_g10|2000|0.009033|0.009033|0.008753|0.844|
|joint_g10|2000|0.008615|0.009129|0.007180|0.770|
|state_match_g10|2000|0.008340|0.008340|0.007892|0.909|
|joint_g25|2000|0.008096|0.008096|0.007312|0.922|
|state_g10|2000|0.007432|0.007504|0.006981|1.173|
|read_g10|2000|0.006940|0.006940|0.006194|0.972|
|read_w10|2000|0.004944|0.004944|0.004320|1.410|
|joint_w25|2000|0.004839|0.004889|0.004338|1.405|
|state_w10|2000|0.004429|0.004477|0.004064|0.934|
|joint_w10|2000|0.003995|0.004039|0.003470|2.552|

Q is not a success probability. Larger Q is better; smaller radial error is better.

## Matched-process response

|Model|Speed response early32|At32 writes|At128 writes|Late radius response|
|---|---:|---:|---:|---:|
|match_shuffle25_5k|0.8591|0.1919|0.0007|-0.0134|
|match_shuffle25|0.5796|0.0860|0.0023|0.0254|
|uncond_w10|0.0069|0.0158|-0.0006|-0.0000|
|read_w01|0.4726|0.0360|-0.0053|0.0062|
|joint_match_g10|0.4109|0.0117|-0.0001|-0.0000|
|joint_g10|0.6524|0.0106|-0.0017|0.0000|
|state_match_g10|0.4409|0.0094|-0.0042|0.0013|
|joint_g25|0.6036|0.0496|-0.0012|-0.0000|
|state_g10|0.7640|0.0271|0.0005|-0.0008|
|read_g10|0.1356|-0.0016|-0.0006|-0.0000|
|read_w10|-0.0043|0.0018|0.0000|-0.0000|
|joint_w25|0.0169|-0.0003|-0.0000|-0.0000|
|state_w10|0.0172|-0.0034|-0.0000|0.0000|
|joint_w10|0.0082|-0.0020|0.0000|0.0000|

Ideal normalized response is1. These are median responses to matched history interventions, not orbit pass rates. Long paths are evaluation-only.
