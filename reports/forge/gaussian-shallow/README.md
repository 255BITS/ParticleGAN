# One hidden layer for the scalar Gaussian

This bounded architecture diagnostic tests whether simplifying both Gaussian
networks from two hidden layers to one makes the original BCAP trainer acquire
the target more reliably. Width32, the critic's two Fourier frequencies, z2,
learned256-location uniform MoG at sigma.1, batch128 and the whole constant-rate
BCAP recipe remain fixed. This arm changes depth independently of the separate
Fourier-removal investigation.

The shared smoke/stability split asks two distinct questions. Tier1 requires a
scheduled full-law pass by1,000 updates, confirmed by an independent draw from
the same trained state. Tier2 requires every stationary check after1,000 through
4,000 to pass, then reacquisition after the target mean shifts2→3 and retention
through6,000. The original five-terminal acquisition grade remains a secondary
readout and all historical evidence retains its original grade.

The shared public GANTrainer runner owns execution; this report introduces a
task variant rather than a new optimizer or copied training loop. The reservation
is exactly6,000 new updates,120 smoke plus600 stability seconds,seed0 and zero
scientific retries. Oracle/scorer, restore and media checks are software evidence.
Raw stdout, observations and full checkpoints stay under ignored `runs/api/`;
only compact final metrics, provenance, reproduction inputs and actual-training
GIFs are published.

The smaller generator has129 parameters and the Fourier critic225, versus1,185
and1,281 in the original depth2 host. Public named deterministic initialization
remains the policy. Changed tensor shapes necessarily produce different network
tensors; the prior coordinates, sampling law and target batch streams remain
matched. This does not silently substitute a target-informed affine fixture.

[Archived controls](controls.json) preserve original BCAP recipe, source and
receipt identities with zero new cost. Those results do not supply qualification
under the revised smoke gate. The current goal comparison remains the repository's
single generated [technique inventory](../technique-inventory.md).

The protocol and numerical source must be frozen before execution. Results and
an adoption recommendation will be added after this declared experiment finishes.
