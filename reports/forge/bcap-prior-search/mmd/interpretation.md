# Initial-distribution MMD: completed search

The registered MMD bracket repaired none of the BCAP leader's 12 Tier 2 failures.
The matched control passed all six required Tier 1 tasks and 9/21 Tier 2 tasks.
Weight `.1` passed Tier 1 and completed all 21 Tier 2 tasks, passing eight and
regressing word hold. Weights `1` and `10` passed 5/6 and 4/6 Tier 1 tasks;
their 21 Tier 2 cells are blocked and unmeasured. Both retained Tier 3 tasks
are outside this Tier 2 search. Actual request order was `1, .1, 10`, bound to
each request's recipe and each trial's settings.

Keep leader defaults. This bounded initial-distribution tether supplies no
qualification gain in the registered bracket. The twelve original Tier 2
failures persist: Gaussian stability, mode hold, unequal-mass/unequal-width/
anisotropic/overlapping vectors, bars/blobs/intensity images, and all three
100-Gaussian tasks. The [full readout](README.md) lists every measurement with
its frozen score/target; component diagnostics carry **No declared gate**
when the protocol defines only an aggregate threshold. A component's expected
data mass is a data-law reference, not an additional numerical gate.

Word `.1` hold illustrates why an endpoint alone is insufficient. It restores
its own independently certified prefix at step 1,667, adds the declared 4,000
updates, and ends at 5,667 with passing endpoint metrics. Only 24/25 mandatory
primary and independent confirmation checks pass. At original step 2,001,
both cohorts have reconstruction exactness `0 / == 1` and minimum token
probability `.19466007 / >= .9`. Generated quality `1 / >= .95`, modes
`5 / == 5`, and primary/confirmation mass TV `.03671875/.02890625 / <= .1`
pass. Its restored prefix/checkpoint/state identities are retained in
[audit.json](audit.json); the common readout retains both original failure
packet hashes and whole observation hashes. The leader's own prefix is 834;
this task deliberately restores each recipe's own earliest certified state.

Increasing the weight produces earlier regressions. Ring16 endpoint
all-component covariance error is `1.434798 / <= .85` at weight `1` and
`5.332069 / <= .85` at weight `10`. Weight `.1` reaches
`.546139 / <= .85`, with HQ `.944092 / >= .85`, 16 modes and an 11-check
passing suffix. AE/GAN hold at weight `10` ends with passing reconstruction
MSE `.005111 / <= .05` and hold `.026898 / <= .35`, but original step 219 has
MSE `.082680 / <= .05`, leaving only `3 / >= 5` terminal passing checks.
[Original terminal-window witnesses](failure-witnesses.md) retain the frozen
gates and original packet hashes for every failed sustained transfer task.

The 100-Gaussian tasks remain far from their full gates. Weight `.1` original
live holdout precision is `.24413/.23907/.34425 / >= .97` on grid/rotated/
staggered100; independently recomputed coverage from those same saved draws
gives precision `.24409/.23905/.34425` and modes `11/12/22 / == 100`.
Keep the original live-accuracy and recomputed-coverage cohorts distinct.
Staggered100 improves some diagnostics over the control but still fails
coverage, sustained accuracy and the joint holdout gate. Unavailable center,
covariance and radial accuracy scalars remain explicitly unavailable.

The objective compares a deterministic, at-most-256-row current panel to an
original fixed initial panel, using three fixed IMQ bandwidths scaled by the
initial population covariance trace. This is a finite-panel empirical tether;
its gradients cover the selected current panel rather than the full empirical
law. Actual prior motion requires panel rows that are also optimizer-owned
sampled rows. The leader updates each such row by
`-.030 * g / sqrt(||g||² + .001²)`, including adversarial and weighted MMD
gradients. Coefficients change direction and nonlinear smoothed saturation;
they do not give proportional changes in step size. The results therefore
describe this objective, panel rule and bracket under the fixed leader.

The [saved-evidence audit](audit.json) verifies 1,208 executed scientific source
files against commit `e85cabc484140cd8afa3c44ce2abe1b93daf0443` and source
digest `4a4dd33ea938ea6a48f7c6c5faa83c1237e87f41b28d2848e51dbdf99847d04e`,
original non-superseded certificates, 86 immutable artifact manifests, declared
recipe deltas, initial parameter/prior receipts, available saved initial
tensors, named streams, update counts and frozen budgets. Word continuation
retains its own prefix length. The diagnostic clock probe grants no
qualification credit. All submissions are terminal, with no running jobs,
paused campaigns or outstanding reservation.

Charged cost is 2,591.662 control seconds plus 4,253.125 search seconds:
6,844.787 seconds total. This includes the original infrastructure cancellation
at 398.778 charged seconds (398.362 elapsed seconds). GPU context contention
was repaired with NVIDIA MPS under eight exact matching numerical probe
digests, unchanged 900-second/20,001-update word allowances, unchanged
source/recipe/seed/gates, and preserved interrupted attempts. Existing repaired
word workers completed unchanged; subsequent new jobs used one training job
per physical GPU across all five queues. Cross-control word scientific states,
named streams and consumed fixture data agree at steps 834 and 20,001; caller
global CPU/CUDA snapshots differ. Costs are execution accounting rather than a
speed ranking. No scientific retries, new strengths or seed variants occurred.

[All 70 certified task GIFs and receipts](media/README.md) are exported solely
from original saved actual-training observations, including PASS, FAIL and
clock diagnostic tasks. The exporter forbids live model construction/forward,
training and sampling; it added zero optimizer updates, draws or rescoring.
The existing [current leaderboard](../../technique-inventory.md) remains the
single goal leaderboard. This is bounded tuning evidence with no independent
confirmation study or default adoption.

Reproduce the read-only publication from the original archive:

```sh
PYTHONPATH=. python reports/forge/bcap-prior-search/workflow.py publish --name mmd --archive /mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/mmd
PYTHONPATH=. python reports/forge/bcap-prior-search/mmd/audit.py --archive /mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/mmd --control-word-audit reports/forge/bcap-prior-search/control_word_audit.json
PYTHONPATH=. python reports/forge/bcap-prior-search/export_saved_media.py --name mmd --archive /mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/mmd
```

Saved-only reproduction commands require the [mmd feature's executed checkout](https://github.com/255BITS/ParticleGAN/tree/e85cabc484140cd8afa3c44ce2abe1b93daf0443) and the original local archive. This aggregate branch contains reports and saved evidence; the feature PR contains its implementation.
