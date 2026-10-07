Ring16's full polar optimizer amplifies an update-401 hidden-critic gradient
difference of 1.03e-7 into a .252 direction difference. The frozen CUDA probe
shows smooth spectral damping reduces that discrepancy about 1,288-fold.

Two fresh-live GPU trials compare the same damping rule once at 401 against
every update through 1600. The boundary arm verifies the unchanged 400 prefix
and ends with 31 consecutive full passes (covariance .458855); its single
independent first-pass confirmation narrowly fails (.859541 > .85), so confirmed
smoke is FAIL. Every-step damping never passes (final covariance 1.057656).
Both preserve constant rates, seed 0, public initialization, MoG prior, batch,
sampling law and full bounds. Candidates never reload the archive.

Validation: both 1600-update arms complete on RTX A6000/CUDA 13.0; 3200 new
updates, 45.35 training seconds, 193 scored draws, zero retries. Frozen bindings,
exact boundary prefix, matched target batches, artifact hashes, call counts,
budgets and actual-training GIF provenance pass verification. Bulk evidence is
archived outside Git with exact SHA/bytes/member receipts. The blocked
preparation and PR331 evidence retain their original identities. Production
defaults and qualifications remain unchanged.

See reports/forge/ring16-damping/README.md, results.json,
execution-verification.json, archive.json and the two actual-training GIFs.
