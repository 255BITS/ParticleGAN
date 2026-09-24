# Finite-GH9 whole-map free-output screen

This separate source epoch passed its declared **pure-output** screens. It
does not yet change or qualify ParticleGAN training. The target is the
cumulative native real-bank empirical law convolved with the same frozen
`h=.031286240422040236`. GH5 proposes at most twelve replacements from the
*current* real128 bank and at most twenty EM steps. Only the completed cloud
is materialized: accept it if the **finite GH9** cumulative cross-entropy
strictly decreases; otherwise take one GH9 EM step from the original cloud
if it strictly decreases, or rest exactly. No quality label, target mode
center, or target shift enters this update. The original native output-noise
clock is preserved. This is a finite quadrature objective, not a certified
continuous-KL objective.

| Predeclared screen | Result |
| --- | --- |
| Warm1324, 16 updates | 16/16 at eight modes; minimum fixed-late HQ `.999756` |
| Cold1, 16 updates | 16/16 at eight modes/HQ `1`; last five pass |
| Paired ordinary vs one conditionally omitted D bank | Both eight modes/HQ `1`; paired data RNG hashes equal |
| Two conditionally omitted banks, then ordinary banks | Seven modes during omission; first ordinary bank restored eight/HQ `1`; terminal five pass |
| Clean output x-bias `+.35`, same target | First updated checkpoint1340 restored eight/HQ `1`; all five pass; paired frozen arm stays at observed zero modes/HQ `0` |
| Whole finite-GH9 objective | Every selected update strictly decreased it; global Torch RNG unchanged |

The conditional omission probes a missing minibatch while later banks again
come from the original fixed target. The first-bank real samples and initial
GH5/GH9 objectives exactly matched the archived saved-state controls.
All actual selections in these trajectories accepted the completed GH5
proposal; the GH9 one-step fallback and exact rest were exercised by focused
tests, not by these observed arms. Three method tests pass, including the
GH5-no-op case where GH9 still moves and its EM sufficient-decrease
inequality; root's chunked donor helper passes two tests. An independent
source review found no formula or restoration blocker.

The screen took about 246 seconds on one CPU thread. Direct output moves
were large: maximum particle displacement `6.086` warm, `3.068` cold,
`6.278` on the first error-response update. Thus the neural G/prior
parameterization and optimizer may fail to realize these targets. Final
external mode-mass TV is `1/6`, so eight-mode HQ is not exact distribution
match. Current-bank donor restriction also has a separate fixed-target
rest-trap counterexample under audit; this screen does **not** resolve that
acquisition limitation. No neural or long-hold run was launched.

Frozen source SHA-256 `adc5d6f6ded8e6cf297ee90aa0d540723becc632a601b93b5b9a4393cd344485`.
The raw rows, declaration, source bytes, transcript, and verified manifest
are in [`round8-forward-kl-gh9-stress`](continuous-evidence/round8-forward-kl-gh9-stress/manifest.json).
