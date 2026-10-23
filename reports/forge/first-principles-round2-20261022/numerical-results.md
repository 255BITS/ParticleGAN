# First-principles successor comparison

Generated from certified original receipts by [publish.py](publish.py). This research panel does not change ordinary qualification or the [goal leaderboard](../technique-inventory.md).

Executed source `a03373da3421303b89c99a39898b091e22910134`, manifest `668914920afda7bb6fc53d9c918aac92a9c9cd5ae582884fb9f068ff1b8249e8`; protocol seed 0. All task gates and allowances are unchanged.

| Arm | Original failures repaired | Passing controls | New regressions | Paid attempt seconds |
| --- | ---: | ---: | ---: | ---: |
| Original control | 0/11 | 4/4 | 0 | 1395.130 |
| Energy control | 2/11 | 4/4 | 0 | 2206.040 |
| Secant mechanics | 3/11 | 2/4 | 2 | 11637.005 |
| Bregman game | 4/11 | 2/4 | 2 | 1975.078 |

Reserved cap: 112,080 seconds; total paid attempt time: 17213.254 seconds. This is not a speed comparison.

## All task verdicts

Each verdict links to an actual-training GIF rendered only from retained scored observations. No training, sampling or rescoring was added.

| Task | Original control | Energy control | Secant mechanics | Bregman game |
| --- | --- | --- | --- | --- |
| `gaussian1d_smoke` | [PASS](media/bispf-optimized-quintic-v1/gaussian1d_smoke.gif) | [PASS](media/bispf-output-energy-v1/gaussian1d_smoke.gif) | [PASS](media/bispf-secant-extragradient-v1/gaussian1d_smoke.gif) | [PASS](media/bispf-bregman-game-v1/gaussian1d_smoke.gif) |
| `gaussian1d_stability` | [FAIL](media/bispf-optimized-quintic-v1/gaussian1d_stability.gif) | [FAIL](media/bispf-output-energy-v1/gaussian1d_stability.gif) | [FAIL](media/bispf-secant-extragradient-v1/gaussian1d_stability.gif) | [FAIL](media/bispf-bregman-game-v1/gaussian1d_stability.gif) |
| `vector_unequal_mass` | [FAIL](media/bispf-optimized-quintic-v1/vector_unequal_mass.gif) | [FAIL](media/bispf-output-energy-v1/vector_unequal_mass.gif) | [FAIL](media/bispf-secant-extragradient-v1/vector_unequal_mass.gif) | [FAIL](media/bispf-bregman-game-v1/vector_unequal_mass.gif) |
| `vector_unequal_width` | [FAIL](media/bispf-optimized-quintic-v1/vector_unequal_width.gif) | [FAIL](media/bispf-output-energy-v1/vector_unequal_width.gif) | [FAIL](media/bispf-secant-extragradient-v1/vector_unequal_width.gif) | [FAIL](media/bispf-bregman-game-v1/vector_unequal_width.gif) |
| `vector_anisotropic` | [FAIL](media/bispf-optimized-quintic-v1/vector_anisotropic.gif) | [PASS](media/bispf-output-energy-v1/vector_anisotropic.gif) | [PASS](media/bispf-secant-extragradient-v1/vector_anisotropic.gif) | [FAIL](media/bispf-bregman-game-v1/vector_anisotropic.gif) |
| `vector_overlap` | [FAIL](media/bispf-optimized-quintic-v1/vector_overlap.gif) | [FAIL](media/bispf-output-energy-v1/vector_overlap.gif) | [PASS](media/bispf-secant-extragradient-v1/vector_overlap.gif) | [PASS](media/bispf-bregman-game-v1/vector_overlap.gif) |
| `img_bars4` | [FAIL](media/bispf-optimized-quintic-v1/img_bars4.gif) | [FAIL](media/bispf-output-energy-v1/img_bars4.gif) | [FAIL](media/bispf-secant-extragradient-v1/img_bars4.gif) | [PASS](media/bispf-bregman-game-v1/img_bars4.gif) |
| `img_blobs4` | [FAIL](media/bispf-optimized-quintic-v1/img_blobs4.gif) | [PASS](media/bispf-output-energy-v1/img_blobs4.gif) | [PASS](media/bispf-secant-extragradient-v1/img_blobs4.gif) | [PASS](media/bispf-bregman-game-v1/img_blobs4.gif) |
| `img_intensity2` | [FAIL](media/bispf-optimized-quintic-v1/img_intensity2.gif) | [FAIL](media/bispf-output-energy-v1/img_intensity2.gif) | [FAIL](media/bispf-secant-extragradient-v1/img_intensity2.gif) | [PASS](media/bispf-bregman-game-v1/img_intensity2.gif) |
| `grid100` | [FAIL](media/bispf-optimized-quintic-v1/grid100.gif) | [FAIL](media/bispf-output-energy-v1/grid100.gif) | [FAIL](media/bispf-secant-extragradient-v1/grid100.gif) | [FAIL](media/bispf-bregman-game-v1/grid100.gif) |
| `rotated100` | [FAIL](media/bispf-optimized-quintic-v1/rotated100.gif) | [FAIL](media/bispf-output-energy-v1/rotated100.gif) | [FAIL](media/bispf-secant-extragradient-v1/rotated100.gif) | [FAIL](media/bispf-bregman-game-v1/rotated100.gif) |
| `staggered100` | [FAIL](media/bispf-optimized-quintic-v1/staggered100.gif) | [FAIL](media/bispf-output-energy-v1/staggered100.gif) | [FAIL](media/bispf-secant-extragradient-v1/staggered100.gif) | [FAIL](media/bispf-bregman-game-v1/staggered100.gif) |
| `ring16_acquisition` | [PASS](media/bispf-optimized-quintic-v1/ring16_acquisition.gif) | [PASS](media/bispf-output-energy-v1/ring16_acquisition.gif) | [FAIL](media/bispf-secant-extragradient-v1/ring16_acquisition.gif) | [FAIL](media/bispf-bregman-game-v1/ring16_acquisition.gif) |
| `mode_hold` | [PASS](media/bispf-optimized-quintic-v1/mode_hold.gif) | [PASS](media/bispf-output-energy-v1/mode_hold.gif) | [FAIL](media/bispf-secant-extragradient-v1/mode_hold.gif) | [FAIL](media/bispf-bregman-game-v1/mode_hold.gif) |
| `img_stripes2` | [PASS](media/bispf-optimized-quintic-v1/img_stripes2.gif) | [PASS](media/bispf-output-energy-v1/img_stripes2.gif) | [PASS](media/bispf-secant-extragradient-v1/img_stripes2.gif) | [PASS](media/bispf-bregman-game-v1/img_stripes2.gif) |

## Numerical detail

Endpoint metrics describe the last state; they do not replace sustained gates. Gaussian smoke is an independently confirmed acquisition gate and may PASS with a failing endpoint. The [compact JSON](comparison.json) retains every final metric, evaluator summary, attempt identity and original receipt hash.

### `gaussian1d_smoke`

- **Original control: PASS** — first confirmed step 500; endpoint KS 0.135031 (acquisition gate).
- **Energy control: PASS** — first confirmed step 209; endpoint KS 0.292791 (acquisition gate).
- **Secant mechanics: PASS** — first confirmed step 125; endpoint KS 0.036120 (acquisition gate).
- **Bregman game: PASS** — first confirmed step 167; endpoint KS 0.207931 (acquisition gate).

### `gaussian1d_stability`

- **Original control: FAIL** — stationary 6/72; shifted hold 2/24; reacquisition FAIL.
- **Energy control: FAIL** — stationary 12/72; shifted hold 0/24; reacquisition FAIL.
- **Secant mechanics: FAIL** — stationary 40/72; shifted hold 12/24; reacquisition FAIL.
- **Bregman game: FAIL** — stationary 10/72; shifted hold 2/24; reacquisition FAIL.

### `vector_unequal_mass`

- **Original control: FAIL** — mass_tv 0.057422; min_mass_ratio 0.000000; component_covariance_error 1.767639; component_min_eigen_ratio 0.000000; covariance_error 0.138733; mean_error 0.132772; sw1_normalized 0.101744; hq 0.977295; terminal suffix 0 (required 5).
- **Energy control: FAIL** — mass_tv 0.059180; min_mass_ratio 0.546499; component_covariance_error 0.337878; component_min_eigen_ratio 0.266615; covariance_error 0.161179; mean_error 0.123653; sw1_normalized 0.093137; hq 0.984863; terminal suffix 1 (required 5).
- **Secant mechanics: FAIL** — mass_tv 0.084570; min_mass_ratio 0.413161; component_covariance_error 0.402759; component_min_eigen_ratio 0.062121; covariance_error 0.218017; mean_error 0.121458; sw1_normalized 0.089105; hq 0.996826; terminal suffix 0 (required 5).
- **Bregman game: FAIL** — mass_tv 0.107109; min_mass_ratio 0.000000; component_covariance_error 1.876737; component_min_eigen_ratio 0.000000; covariance_error 0.272815; mean_error 0.090183; sw1_normalized 0.163807; hq 0.739258; terminal suffix 0 (required 5).

### `vector_unequal_width`

- **Original control: FAIL** — mass_tv 0.277588; min_mass_ratio 0.250000; component_covariance_error 9.686903; component_min_eigen_ratio 0.093470; covariance_error 0.225306; mean_error 0.398091; sw1_normalized 0.287383; hq 0.981934; terminal suffix 0 (required 5).
- **Energy control: FAIL** — mass_tv 0.089355; min_mass_ratio 0.814453; component_covariance_error 0.716421; component_min_eigen_ratio 0.180936; covariance_error 0.034062; mean_error 0.120573; sw1_normalized 0.091365; hq 0.994629; terminal suffix 1 (required 5).
- **Secant mechanics: FAIL** — mass_tv 0.084961; min_mass_ratio 0.815430; component_covariance_error 0.464316; component_min_eigen_ratio 0.649668; covariance_error 0.098491; mean_error 0.136730; sw1_normalized 0.097175; hq 0.977539; terminal suffix 2 (required 5).
- **Bregman game: FAIL** — mass_tv 0.006592; min_mass_ratio 0.980469; component_covariance_error 0.488562; component_min_eigen_ratio 0.147383; covariance_error 0.050238; mean_error 0.017895; sw1_normalized 0.045033; hq 0.999268; terminal suffix 0 (required 5).

### `vector_anisotropic`

- **Original control: FAIL** — mass_tv 0.175212; min_mass_ratio 0.533936; component_covariance_error 0.313814; component_min_eigen_ratio 0.325150; covariance_error 0.231226; mean_error 0.278775; sw1_normalized 0.213624; hq 0.995605; terminal suffix 0 (required 5).
- **Energy control: PASS** — mass_tv 0.053385; min_mass_ratio 0.907471; component_covariance_error 0.167197; component_min_eigen_ratio 0.473108; covariance_error 0.082343; mean_error 0.070056; sw1_normalized 0.059956; hq 0.979492; terminal suffix 6 (required 5).
- **Secant mechanics: PASS** — mass_tv 0.099202; min_mass_ratio 0.702393; component_covariance_error 0.125743; component_min_eigen_ratio 0.411338; covariance_error 0.211207; mean_error 0.191719; sw1_normalized 0.126289; hq 0.989990; terminal suffix 18 (required 5).
- **Bregman game: FAIL** — mass_tv 0.080485; min_mass_ratio 0.795410; component_covariance_error 0.911936; component_min_eigen_ratio 0.004879; covariance_error 0.150791; mean_error 0.135622; sw1_normalized 0.128672; hq 0.941895; terminal suffix 0 (required 5).

### `vector_overlap`

- **Original control: FAIL** — covariance_error 0.522296; mean_error 0.101512; sw1_normalized 0.149086; terminal suffix 0 (required 5).
- **Energy control: FAIL** — covariance_error 0.150126; mean_error 0.040615; sw1_normalized 0.052610; terminal suffix 1 (required 5).
- **Secant mechanics: PASS** — covariance_error 0.124908; mean_error 0.059381; sw1_normalized 0.049873; terminal suffix 5 (required 5).
- **Bregman game: PASS** — covariance_error 0.259825; mean_error 0.059326; sw1_normalized 0.080468; terminal suffix 9 (required 5).

### `img_bars4`

- **Original control: FAIL** — modes 2; clean 0.96875; terminal suffix 0 (required 5).
- **Energy control: FAIL** — modes 3; clean 0.59375; terminal suffix 0 (required 5).
- **Secant mechanics: FAIL** — modes 3; clean 0.84375; terminal suffix 0 (required 5).
- **Bregman game: PASS** — modes 4; clean 0.93750; terminal suffix 6 (required 5).

### `img_blobs4`

- **Original control: FAIL** — modes 3; clean 0.81250; terminal suffix 0 (required 5).
- **Energy control: PASS** — modes 4; clean 0.96875; terminal suffix 8 (required 5).
- **Secant mechanics: PASS** — modes 4; clean 1.00000; terminal suffix 7 (required 5).
- **Bregman game: PASS** — modes 4; clean 0.96875; terminal suffix 8 (required 5).

### `img_intensity2`

- **Original control: FAIL** — modes 2; clean 0.59375; terminal suffix 0 (required 5).
- **Energy control: FAIL** — modes 1; clean 0.50000; terminal suffix 0 (required 5).
- **Secant mechanics: FAIL** — modes 2; clean 0.81250; terminal suffix 0 (required 5).
- **Bregman game: PASS** — modes 2; clean 1.00000; terminal suffix 9 (required 5).

### `grid100`

- **Original control: FAIL** — holdout precision 0.30497; mass TV 0.17028; center max 2.8764 sigma; covariance RMS 0.61016; terminal passes 0/5.
- **Energy control: FAIL** — holdout precision 0.32545; mass TV 0.15737; center max 2.3382 sigma; covariance RMS 0.56445; terminal passes 0/5.
- **Secant mechanics: FAIL** — holdout precision 0.62864; mass TV 0.10209; center max 1.0124 sigma; covariance RMS 1.02872; terminal passes 0/5.
- **Bregman game: FAIL** — holdout precision 0.07912; mass TV 0.03085; center max undefined sigma; covariance RMS undefined; terminal passes 0/5.

### `rotated100`

- **Original control: FAIL** — holdout precision 0.22301; mass TV 0.11132; center max undefined sigma; covariance RMS undefined; terminal passes 0/5.
- **Energy control: FAIL** — holdout precision 0.24826; mass TV 0.09887; center max 2.2934 sigma; covariance RMS 0.65741; terminal passes 0/5.
- **Secant mechanics: FAIL** — holdout precision 0.60551; mass TV 0.08264; center max 0.8903 sigma; covariance RMS 1.05377; terminal passes 0/5.
- **Bregman game: FAIL** — holdout precision 0.06664; mass TV 0.06279; center max undefined sigma; covariance RMS undefined; terminal passes 0/5.

### `staggered100`

- **Original control: FAIL** — holdout precision 0.33617; mass TV 0.12142; center max 2.5165 sigma; covariance RMS 0.70895; terminal passes 0/5.
- **Energy control: FAIL** — holdout precision 0.29757; mass TV 0.11485; center max undefined sigma; covariance RMS undefined; terminal passes 0/5.
- **Secant mechanics: FAIL** — holdout precision 0.63868; mass TV 0.08418; center max 0.4821 sigma; covariance RMS 0.97611; terminal passes 0/5.
- **Bregman game: FAIL** — holdout precision 0.11210; mass TV 0.04194; center max undefined sigma; covariance RMS undefined; terminal passes 0/5.

### `ring16_acquisition`

- **Original control: PASS** — mass_tv 0.092041; min_mass_ratio 0.671875; component_covariance_error 0.466252; component_min_eigen_ratio 0.380435; covariance_error 0.081625; mean_error 0.041858; sw1_normalized 0.041541; hq 0.951904; terminal suffix 12 (required 5).
- **Energy control: PASS** — mass_tv 0.061279; min_mass_ratio 0.675781; component_covariance_error 0.486425; component_min_eigen_ratio 0.342219; covariance_error 0.055473; mean_error 0.077457; sw1_normalized 0.056838; hq 0.961914; terminal suffix 47 (required 5).
- **Secant mechanics: FAIL** — mass_tv 0.054688; min_mass_ratio 0.714844; component_covariance_error 1.087105; component_min_eigen_ratio 0.692716; covariance_error 0.039469; mean_error 0.071022; sw1_normalized 0.051792; hq 0.911621; terminal suffix 0 (required 5).
- **Bregman game: FAIL** — mass_tv 0.024658; min_mass_ratio 0.902344; component_covariance_error 1.385529; component_min_eigen_ratio 0.592251; covariance_error 0.063915; mean_error 0.042420; sw1_normalized 0.033709; hq 0.844971; terminal suffix 0 (required 5).

### `mode_hold`

- **Original control: PASS** — hq 0.969238; terminal suffix 10 (required 5).
- **Energy control: PASS** — hq 0.986328; terminal suffix 7 (required 5).
- **Secant mechanics: FAIL** — hq 0.904297; terminal suffix 3 (required 5).
- **Bregman game: FAIL** — hq 0.315430; terminal suffix 0 (required 5).

### `img_stripes2`

- **Original control: PASS** — modes 2; clean 1.00000; terminal suffix 12 (required 5).
- **Energy control: PASS** — modes 2; clean 1.00000; terminal suffix 5 (required 5).
- **Secant mechanics: PASS** — modes 2; clean 1.00000; terminal suffix 5 (required 5).
- **Bregman game: PASS** — modes 2; clean 1.00000; terminal suffix 11 (required 5).

## Binding checks

Verified 45 completed changed-arm/control task pairs; 0 pairs have unavailable or unequal completed training. Verified pairs have identical task declarations, initial model hashes and priors, sampling laws, update counts and final consumed non-evaluation RNG stream hashes. Gaussian data batches additionally have direct sequence hashes. Other tasks verify their fixed data law and identical consumed named streams; no direct batch fingerprint is claimed for those adapters. Evaluation draws retain their own isolated streams.

Gaussian continuation starts from each arm's own terminal 1,000-update smoke checkpoint, after its acquisition gate passes; that terminal state can fail output bounds. Matching initialized model identities refer to the common public initialization contract, not equal trained continuation states.

All admitted requests share the same complete source manifest, runtime and hardware profile. Raw stdout, full readouts, per-update events, tensors and checkpoints remain in the artifact archive named in [protocol.json](protocol.json). Compact provenance includes their exact hashes. Historical selected results keep their original source and qualification identity.

The fresh control exactly reproduces 15/15 archived final metric dictionaries and gate verdicts. This numerical agreement does not merge evidence cohorts. The original full-readout SHA-256 is `50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe` from source `d9d72d53761c898bce20102e1995fc3d899c1447`.
