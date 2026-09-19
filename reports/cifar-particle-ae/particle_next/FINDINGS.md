# Completed count and duration training

New best observed FID50k: **15.7527**, 16,384 particles at80k steps. The 32,768-particle arm ends40k at16.3451 and improves the matched16k40k endpoint by0.7532. Target13 remains2.7527 below the new best. These runs support both additional training and another count doubling, without establishing a scaling law or training-run uncertainty. No repeat seeds were used.

## Endpoint leaderboard

| Particles | Steps | FID50k | Density | Coverage |
|---|---:|---:|---:|---:|
|16,384|80,000|15.7527|0.6469|64.26%|
|32,768|40,000|16.3451|0.6168|62.17%|
|16,384|40,000|17.0982|0.6160|62.17%|
|4,096|40,000|17.2350|0.6148|61.39%|
|8,192|40,000|17.7683|0.5956|59.15%|

Different duration endpoints are shown explicitly;16k80k is not a matched-count comparison with32k40k. Best sampled32k point is16.2066 at35k, versus16.3451 at40k. Previous overall best was4k35k16.5033. Neither new trajectory is monotonic.

## Trajectories

16k at40/50/60/70/80k:17.0982,16.5965,17.4222,16.8014,15.7527. The last point improves1.3455 over its starting endpoint and0.7506 over the previous overall best. Density improves0.6160→0.6469 and coverage62.17%→64.26%, supporting the FID improvement. Sibling information rises2.6886→3.1149 of4 available bits.

32k at15/20/25/30/35/40k:18.5834,17.3152,16.9717,16.3545,16.2066,16.3451. It is worse than16k at15k, then better at every matched20–40k evaluation. Its40k density/coverage are essentially unchanged from16k40k despite lower FID; particle scaling has not improved every distribution diagnostic. InitialFID19.4478 matches original parent19.4482 within0.0005.

Both endpoint sample grids visually contain varied recognizable objects with remaining shape/detail artifacts. No claim of complete collapse or visual equivalence to real data. Samples are small qualitative panels.

## Cost and checks

16k continuation:40,000 additional updates in29.17 training minutes (22.86 updates/sec). 32k:30,000 updates in22.59 training minutes (22.14 updates/sec), about3.3% more time per update. Evaluation and diagnostic time excluded. Peak GPU memory approximately5.32/5.35GB. Frozen-feature, sigma, learning-rate, checkpoint and source certificates verified by the pipeline; original certified sources unchanged.

## Recommendation

Continue32k from40k to80k, withFID50k every10k and the same endpoint information/density/coverage probe. It has the better matched40k result, while16k demonstrates that training beyond40k can still help. Reuse the completed16k80k curve. This is the most informative next test before increasing to64k; do not infer that32k must win at80k. A longer16k duration arm can separately test whether the late improvement persists toward200k, but the user subsequently authorized the next stage described below.

The current E-only reconstruction objective cannot directly pull G/particles toward pixel averages. More centers and training help, but the mechanism of the earlier plateau remains unresolved. Greater decodable sibling information can include artifacts and is not exact entropy or a semantic-mode count.

## Completed endpoint information and validation

Both training arms and both full endpoint probes completed and passed current-source/config certification. The32k probe uses the exact same real-feature reference tensor as the16k probe.32k40k sibling information3.4838/5 bits (parent SE0.0894);16k80k3.1149/4 (SE0.0783). Compare count at matched40k:16k2.6886/4 versus32k3.4838/5. More bits at32k accompany lowerFID but essentially identical density/coverage, so they do not establish broader real-data coverage. Shuffled-label and identical-clone controls are near zero. Raw per-parent statistics are preserved in respective results.json.

## Authorized follow-up

User asked to continue32k and choose between64k versus longer16k on the other GPU. Chose16k80k→160k because its latest checkpoint is the best and improved density/coverage; longer duration tests persistence toward200k. GPU1 resumes32k40k→80k for the matched80k count comparison. Both useFID50k every10k, full saved optimizer/EMA/RNG, unchanged recipe, and automatic endpoint probes. Launch metadata and plan: `../particle_duration/`. No64k arm or further automatic promotion is queued.
