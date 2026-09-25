# Image GAN solvability: shared configurations

Seed0, unchanged live quality/coverage gates, 24 observations and five final passing checks. Architecture and budget changes are explicit. Supervised controls do not enter GAN ranking. Previously reserved residual bars are now seen development data; no fresh holdout is evaluated.

| Shared card | Scope | Sustained / 4 | Stripes | Bars | Blobs | Intensity | CPU seconds |
| --- | --- | ---: | --- | --- | --- | --- | ---: |
| residual12 | architecture changed, original G/D widths | 2/4 | PASS 2/2 · 100.0% | FAIL 3/4 · 93.8% | FAIL 3/4 · 71.9% | PASS 2/2 · 100.0% | 28.5 |
| transpose16 | width changed, original transpose architecture | 0/4 | FAIL 1/2 · 93.8% | FAIL 2/4 · 81.2% | FAIL 3/4 · 90.6% | FAIL 0/2 · 0.0% | 23.7 |

Each cell reports sustained verdict, final quality-qualified modes and HQ. A final pass with an insufficient passing suffix remains FAIL. Missing tasks cannot form a shared winner.

| Supervised expressivity control (not a GAN) | Result | Confirmed step |
| --- | --- | ---: |

Every exact spec, failure, curve, action trace and source/runtime hash is retained in the episode JSON.gz artifacts.
