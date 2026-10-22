# Measured allocation outcome

**Energy adds two sustained repairs: anisotropic density and corner blobs.**
It passes **6/15** diagnostic tasks versus the matched BISPF control's **4/15**,
retaining Gaussian acquisition, ring16 acquisition, mode hold and stripes. Of
the eleven original failing tasks, two pass and nine remain FAIL. The parent
[comparison](../README.md) owns the single leaderboard; this is a mechanism
analysis, not ordinary qualification or public-default promotion.

Both arms executed source `2a2b662bdfa6249076b27dbea4c136e23d3e7a80`, frozen
digest `8bfcc01cd7482b1b348edb2162567adfdcdaa7d308f75fc834774302c6ac4acb`,
through the public GANTrainer with protocol seed 0 and the public deterministic
initializer. The sole effective recipe delta is `output_energy_weight=1.0`.
The [analysis receipt](analysis-receipt.json) binds the complete local readout
hashes and archived reduction. All fifteen pairs match initial state/prior,
sampling, update counts and complete consumed non-evaluation stream metadata.
Thirty saved provenance checkpoints pass SHA-256 verification before their
controller counters are read. This analysis adds zero model forwards, sampling
draws or optimizer updates. Parent artifact auditing owns complete execution
certification; metadata comparisons alone are not an independent artifact audit.

Anisotropic mass TV improves **.175212 → .053385**, and full component covariance
error **.313814 → .167197**. The candidate passes eight scheduled checks with a
terminal suffix of six, exceeding the required five. The original control had
no complete passing check. Corner blobs improve from three to four modes and
HQ **.8125 → .96875**; all final eight checks pass. Their actual API training
GIFs illustrate the measured goals: [anisotropic](../media/bispf-output-energy-v1/vector_anisotropic.gif)
and [blobs](../media/bispf-output-energy-v1/img_blobs4.gif). Numerical sustained
gates, rather than those images, determine the results.

Allocation improves further without completing every task:

- **Unequal mass:** rare support appears at update 450 and remains present at
  every subsequent observation. Endpoint rare occupancy is **81/4096** versus
  control **0/4096**. All endpoint gates pass, with minimum mass ratio .546499,
  full covariance error .337878 and minimum eigen ratio .266615. Only two of
  24 complete observations pass; the terminal suffix is **1/5**. At updates
  1050/1100/1150 the sole failed scalar gate is minimum component eigen ratio:
  the rare component gives **.129161/.030277/.063941**, below **.15**, while
  retaining **76/95/73** samples. Thus late failure is rare-component shape,
  rather than disappearance. Across the entire curve, covariance and minimum
  eigen ratio each fail 19/24 checks; minimum mass ratio fails 9/24.
- **Unequal width:** all endpoint gates also pass, with full covariance error
  **.716421 ≤ .85**, minimum eigen ratio **.180936 ≥ .15**, and mass TV .089355.
  Only four complete checks pass, with suffix **1/5**. Full covariance fails
  19/24 observations, minimum eigen ratio 4/24, and HQ 3/24. A good terminal
  fit does not satisfy retention.
- **Overlap:** complete passing observations improve **10/24 → 20/24**;
  endpoint covariance error improves **.522296 → .150126**. The suffix remains
  **1/5**. Its four failing checks all fail only mean error: updates
  750/850/900/1150 give **.179081/.195070/.194880/.160809**, above **.15**.
  Covariance and sliced-distance bounds pass even on those observations.

The remaining quality and settling failures are explicit:

- **Bars:** modes improve **2 → 3**, below four; HQ worsens
  **.96875 → .59375**, below .9. Mode count fails 24/24 checks and HQ 17/24.
  **Intensity:** nearest-template fractions are nearly balanced
  (.53125/.46875), but HQ is **.5**, below .9, and only one quality-qualified
  mode survives, below two. Both bounds fail 24/24. Population correction
  alone cannot certify clean images.
- **Gaussian stability:** stationary passing checks improve **6 → 12 of 72**,
  while shifted hold worsens **2 → 0 of 24**; all 72/24 are required and
  deadline reacquisition also fails. At update 6000, mean and scale meet their
  bounds but **KS .065143 > .05**. Both smoke acquisition gates remain PASS;
  their update-1000 endpoint KS values are control **.135031** and energy
  **.292791**. Smoke permits earlier confirmed acquisition and cannot certify
  the subsequent fit.
- **Native100:** all three remain FAIL, with **0/5** complete final checks
  each. Independent 100k-draw holdout precision is grid **.32545**, rotated
  **.24826**, staggered **.29757**, versus **.97** required. Their accuracy
  mass TV values **.15737/.09887/.11485** exceed **.06**. Grid/rotated accepted
  centre RMS errors **1.40422/1.34715** exceed **.20**, absolute covariance
  trace biases **.34754/.46799** exceed **.10**, and radial KS
  **.43963/.43833** exceed **.04**. Staggered's conditional shape statistics
  are undefined from insufficient accepted coverage; they are not successful
  shape measurements. Native broad mass gains cannot replace these gates.

All four control passes survive. Ring16's terminal suffix is **47**, mode
hold's **7** and stripes' **5**. These are one global recipe's outcomes; no
task-specific selection, changed sampling law, gate relaxation, seed repeat or
budget extension enters.

Direction protection seldom intervenes on the repaired vectors: conflicts are
**6/1200** unequal mass, **0/1200** width and **3/1200** anisotropic, with zero
stalls. Gaussian's saved counter is **1386/6000** and overlap's **151/1200**.
All fifteen candidate checkpoints show zero Pareto stalls. These are whole
checkpoint counters; Gaussian stability includes its own smoke prefix, so
adding those two counters would double-count that prefix. They establish
intervention frequency, not a causal explanation of the failed checks. Energy
is an uncooled direct output term under the existing adversarial direction
protection; the original critic and SBNP laws otherwise remain unchanged.

This is a seed-0, clean/live diagnostic panel with each task's original
architecture, initial MoG, data sequence and finite budget. It is not a
robustness study or the ordinary whole suite. The finite energy numerator is
unbiased before normalization; its target-batch normalization makes the ratio
biased and can correlate force strength with real minibatch composition. A
global energy witness also does not bound each component's density. Neither
limitation is established as the cause of these observed failures. Small
analytical software fixtures, including all sixteen allocation checks, are
separate from these actual-training results.

**Recommendation:** retain the two measured repairs and the exact failed
revision's evidence. Do not claim that energy solves allocation, shape and
settling together, and do not launch a weight sweep or an unchanged extension.
There is no new measured force mechanism here that justifies a paid successor.
A next bounded diagnostic can use at most **60 CPU seconds**, zero GPU
reservations, zero optimizer updates and zero draws to independently inspect
the existing mass/width sample tensors at updates 1000/1050/1100/1150/1200,
checking component assignment, covariance eigenvalues and outlier contribution.
It should supply a concrete mechanism before another global trainer delta.

Reproduce this read-only reduction from the repository root:

```sh
python reports/forge/three-theme-repairs-20261022/allocation/reduce.py \
  --control /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/full-readouts/bispf-optimized-quintic-v1.json \
  --candidate /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/full-readouts/bispf-output-energy-v1.json \
  --output /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/allocation-analysis.json \
  --inspect-checkpoints
```

Missing or mismatched original readouts/checkpoints block verification, rather
than authorizing retraining. Bulk readouts, observations and tensors remain in
the local artifact archive; committed files contain the compact interpretation,
input/output identities and reducer source.
