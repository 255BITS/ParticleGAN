#16k retains more identifiable sibling information and beats8k on quality proxies

Both endpoint probes completed and certified; frozencheckpoint/parenthashes unchanged, identical cached real features,10000real/fake images,k5. Existing4096 diagnostic reused. Both GPUs are now idle and the extension pipeline exited successfully. No further job queued.

| Particles at40k | FID50k | Decodable sibling bits | Available bits | Density | Coverage |
|---|---:|---:|---:|---:|---:|
|4096|17.2350|1.4854|2|0.6148|61.39%|
|8192|17.7683|1.9164|3|0.5956|59.15%|
|16384|17.0982|2.6886|4|0.6160|62.17%|

The16k endpoint improves over8k on FID,density andcoverage. Versus4k at40k, the gains are small:0.1368 lowerFID,0.00116 higherdensity and0.78percentage-point highercoverage. These do not establish a large reliable count advantage from one trajectory.4k at35k remains the best observed checkpoint: FID16.5033,density0.6303,coverage62.58%, all stronger than the16k40k endpoint on these measures.

More sibling identity continues to become visible with training.8k bits increase1.7195→1.9164 between20k/40k;16k increases2.2756→2.6886. Conditional decoding uses a restricted classifier and is not exact entropy. New endpoint parentSEs are0.0913/0.0971; all observed per-parent estimates arepositive, with minima1.0262/1.5466. Shuffled-label controls−0.00077/−0.00149 and identical-clone controls~0 support recoverable feature distinctions rather than label leakage.

Information alone is insufficient for quality: the8k run gainsbits while density falls0.6352→0.5956 andcoverage60.29%→59.15%.16k gainsbits andcoverage59.43%→62.17%, whiledensitydeclines0.6294→0.6160. These are feature-space proxies, not human judgments or causal proof. The data support additional representational flexibility with diminishing gains in distribution quality, rather than an unrestricted particle-count scaling law.

Recommend16k as the next duration candidate because its lasttwoFIDmeasurements improve and its endpoint quality proxies lead8k. A bounded40k→80k continuation withFIDevery10k can test whether progress persists. Do not automatically increasecount again or promote200k based solely onbits. User has only requested completion review so far; no nextjob launched.
