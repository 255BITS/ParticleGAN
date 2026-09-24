# Toy100 geometry and capacity constraints

These are fixed-seed, full-budget ablations of the shared 22-problem recipe. The first source epoch (`346b1ed`) changed only the native toy100 geometry or resources; the older 19 retained their frozen host resources and global recipe. Each native result was measured at 7,000 updates with five final 20,000-draw checks and an independent 100,000-draw holdout. A strict pass requires the original coverage gate **and** the accuracy gate at every final check and holdout. The exact declarations and source hashes are in the [geometry manifest](constraints_geometry_v1_manifest.json), [simple-core interaction manifest](constraints_geometry_simple_interaction_manifest.json), and [moment-box manifest](constraints_geometry_moment_manifest.json). Local raw evidence (ignored by Git) is at `artifacts/toy100-constraints-geometry-v1/` and `artifacts/toy100-constraints-geometry-moment-v1/`.

The incumbent's hardcoded uniform `[-5,5]²` prior can be replaced by a uniform box bounded by the minimum and maximum of one **unlabeled** initial real batch. That variant passed grid and staggered under the original shared optimizer, but missed rotated's first required checkpoint: HQ was 0.9691 versus the frozen 0.9700 cutoff. Its other four final checks and holdout passed. This shows removal of the manual initial box is viable on two shapes, but **not** a strict three-problem or common-22 success. It does not remove the output-noise scale or Fourier frequency choice.

| Grid100 ablation (one field unless noted) | Strict gate | Final modes | Final HQ | Final mass TV |
| --- | --- | ---: | ---: | ---: |
| Incumbent control | PASS | 100 | .9832 | .0448 |
| Public MLP, normal prior, z=2 | FAIL | 1 | .0212 | .6042 |
| Public MLP, normal prior, z=4 | FAIL | 18 | .3943 | .5086 |
| Affine, normal prior | FAIL | 100 | .9729 | .0983 |
| Affine, normal prior, random G | FAIL | 97 | .9700 | .1169 |
| Affine, fixed-square prior, random G | FAIL | 96 | .9828 | .1098 |
| Affine, empirical min/max box | PASS | 100 | .9850 | .0455 |
| Affine, empirical box, random G | FAIL | 92 | .9453 | .1596 |
| Fourier features 0 | FAIL | 35 | .4304 | .0818 |
| Fourier features 1 | FAIL | 100 | .9624 | .0807 |
| Fourier features 2 | FAIL | 31 | .3332 | .2560 |
| Particles 10,000 | FAIL | 20 | .2093 | .2238 |
| Particles 5,000 | FAIL | 5 | .0935 | .1330 |
| Batch 1,024 | FAIL | 12 | .2071 | .2331 |
| Batch 512 | FAIL | 25 | .2352 | .1521 |
| Critic width 64 | PASS | 100 | .9816 | .0466 |
| Critic width 32 | PASS | 100 | .9800 | .0558 |
| Critic depth 2 | PASS | 100 | .9860 | .0508 |
| Joint 10,000 particles, batch 1,024, critic 64, Fourier 2 | PASS | 100 | .9801 | .0522 |

The joint reduction passed grid while several constituent changes failed alone, so it is an interaction result, not evidence that those individual reductions are safe. Four grid survivors were run on both remaining native problems with identical settings. Empirical box and critic width 64 each passed 2/3 native problems; depth 2 and the joint reduction each passed only grid. Critic width 32 was not promoted because its grid center RMS was .1993σ against a .20σ limit. The original cheap older-toy control passed 6/6. No first-epoch variant earned a fresh 19-case replay or a common-22 claim.

The separately predeclared interaction with a simpler global regularizer (`reg_kappa=1`, `reg_coeff=1`, `prior_reg=0`) kept the same empirical min/max box. It passed the six older screening hosts, then a fresh older 19/19 replay and rotated/staggered accuracy checks (5/5 each, both 100k holdouts). Grid failed all five terminal checks: 79/100 modes, HQ .8436, mass TV .1268 at step 7,000; its holdout likewise failed. The archived episodes regraded without policy or source errors. The recipe interaction therefore remains **21/22 FAIL**, despite the older 19/19 and two native passes.

The follow-up `affine_moment_box_v1` source epoch (`3eb182f`) uses one unlabeled batch on a separate fixed stream and analytically sets each uniform-prior half-width to `√3 ×` that axis's population standard deviation, centered on its sample mean. `initialization-samples.npy`, its SHA-256, mean, standard deviation, bounds, and source are archived. The gate independently recomputes the bounds from that retained batch; 121 focused tests pass, including data and bound tampering. Its single predeclared candidate differs from the exact simpler global recipe only by name and native model policy. The resulting grid bounds were x [−4.970, 4.927], y [−4.924, 4.981].

The older six screen passed 6/6, but grid failed all five final checks: 98/100 modes, HQ .9054, mass TV .0961, center RMS .6103σ, radial KS .0590 at step 7,000. The independent 100k holdout also failed (HQ .9042, TV .0968). No rotated, staggered, or full-19 promotion was justified. All 67 source-bound files (11,416,796 bytes) were copied to the ignored local `artifacts/toy100-constraints-geometry-moment-v1/`, byte-for-byte verified with SHA-256 against RAM, and independently regraded at the relocated path: old six PASS, grid coverage and accuracy FAIL. The result shows that even an analytically matched, data-derived box is not robust enough under this exact global core at the frozen seed.

The fixed seed, task geometry, scale, architecture, and budget remain part of this evidence. A passing ablation on one problem is not a recommendation for arbitrary datasets or a substitution for the complete 22-problem gate.
