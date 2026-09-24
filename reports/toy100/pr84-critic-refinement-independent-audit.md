# Independent audit: PR84 empirical critic refinement

This is a scratch diagnostic for the fixed-target `mode_hold` host, not a production or all-host result. It preserves the scheduled shared prefix through update 1,000. Each branch then inherits the same weights, Adam moments, EMA, data stream, and noise streams. The refinement branch uses constant applied rates for updates 1,001 onward: D and network G `0.00425`, prior `0.0085`. The archived [warm evidence](continuous-evidence/pr84-critic-refinement-independent-audit/warm/manifest.json) contains the raw JSON for every branch, declarations, exact source bytes, and SHA-256 checksums.

| Branch | Live checks 1,001–1,200 | Minimum HQ | Final modes/HQ |
| --- | ---: | ---: | ---: |
| Scheduled identity | 200/200 | 0.99634 | 8 / 1.00000 |
| Constant Adam | 6/200 | 0.00537 | 6 / 0.50464 |
| Original PR84 smoothing and curvature | 200/200 | 0.97021 | 8 / 1.00000 |
| Empirical critic refinement | **200/200** | **0.99707** | **8 / 1.00000** |

The source declaration, live source, and archived source hashes agree for all 14 declared files. The four branches share warm-state SHA-256 `6cc79b6e0d11eafae176b68e7d9d8c26c02c886866370134cd70c864fe882e21` and identical noise receipts. The identity branch matches the separately run scheduled cold control. The original PR84 branch exactly matches the earlier independent PR84 warm archive in final state, observations, noise, all 220 diagnostics, and all 200 comparable update records. Refinement and original have constant applied rate ranges at the values above; both optimizer moment counters finish at update 1,200. The refinement receipt verifies 400/400 game-block RNG replays and 200/200 bank and fit RNG checks. No elapsed-time rate decay or quality-based selection is in the active refinement update.

This extra optimization is expensive. In 200 updates, it made 10,674 L-BFGS discriminator gradient evaluations on 1,024 real/fake pairs each (10,930,176 pair evaluations), plus 200 exact first-bank gradient checks and the ordinary three D and three G phase evaluations per update. The median fit used 51.5 closures; 11 reached the 80-closure cap. It selected the lowest finite *training-bank* penalized D loss, with no held-out or quality oracle. The D curvature bound applies to the original Adam proposal, not the subsequent fitted D change; D Adam moments are retained from that proposal. Bank construction and stencil calculations add forward work beyond the gradient counts.

The preceding exact three-branch saved-state filter passed 44/44 local checks, but neither it nor this 200-update warm result establishes perpetual stationary stability or cold acquisition. The longer same-dataset hold is pending at the time of this commit. Active refinement supports only the late zero-input-noise CPU `mode_hold` host; conditional and cold noisy hosts have not been adapted.
