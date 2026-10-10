# Current paired serving coherence: fixed CPU diagnosis

Read only RA7 checkpoints 500, 1000 and 2000. Use the frozen RA7 package,
the original saved-diagnosis functional forward definition, current saved D,
the saved real FIFO and a private generator cloned from each saved CPU RNG.
Fit the existing real-only chart once per checkpoint. All clean FAST and EMA
rows are evaluated in that same chart; no historical GPU cell IDs are reused.

Measure existing learned support, count partition and real-only topology mass,
paired row group/cell agreement and support contingencies. Test the root's
fixed empirical group-coherence threshold ceil((1-Q)N), with Q=.05, without
choosing a threshold from results. Also evaluate the two crossed G/prior pairs
on the fixed row prefix 0,8,...,1016 (128 rows) to describe pair coupling.
No oracle labels or benchmark quality functions enter this diagnostic.

The existing conditional count formulas are recorded with their unchanged
K+2K+2 family. They are descriptive on dependent clean table rows, and two
model comparisons do not share an inferential Q budget automatically. A new
serving family would require common multiplicity and sampling justification.
Failure to discover mismatch is not positive equivalence. The report will
state a conservative categorical confidence radius derived by a union bound
over all subsets and Hoeffding's inequality, with its iid assumptions explicit.

No CUDA, new emissions, training, optimizer calls, seeds, data, reaction plans,
production edits or forced EMA. Preserve checkpoint tensors, source bytes,
inputs and global RNG. Never restore a saved CUDA RNG stream into a CPU stream.
This is a current-chart empirical diagnostic and prospective design, not a
quality evaluation or a historical replay.
