# Image GAN solvability: shared configurations

Seed0, unchanged live quality/coverage gates, 24 observations and five final passing checks. Architecture and budget changes are explicit. Supervised controls do not enter GAN ranking. Previously reserved residual bars are now seen development data; no fresh holdout is evaluated.

| Shared card | Scope | Sustained / 4 | Stripes | Bars | Blobs | Intensity | CPU seconds |
| --- | --- | ---: | --- | --- | --- | --- | ---: |
| residual16_cap10 | architecture changed; cross-domain cap10 | 3/4 | PASS 2/2 · 100.0% | FAIL 4/4 · 81.2% | PASS 4/4 · 96.9% | PASS 2/2 · 100.0% | 26.7 |

Each cell reports sustained verdict, final quality-qualified modes and HQ. A final pass with an insufficient passing suffix remains FAIL. Missing tasks cannot form a shared winner.

| Supervised expressivity control (not a GAN) | Result | Confirmed step |
| --- | --- | ---: |

Every exact spec, failure, curve, action trace and source/runtime hash is retained in the episode JSON.gz artifacts.
