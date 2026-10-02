# Public API image and five-word variants

These are new runnable definitions for all 39 retained image entries (34 ordered banks) and the five-word source demo. `api_images.list_cases()` declares 80 hosts: the 69 recorded image arm architectures, four original checked-in transpose12 hosts, six real conditional variants, and one joint word autoencoder. Every original receipt, rating and GIF remains unchanged.

The unconditional hosts use `GANTrainer`. Conditional hosts use the public recipe optimizers/loss/penalty and `UpdatePolicy`, with actual observed pixels and masks supplied to both G and D. Independent-row evidence and birth/death are explicitly disabled on these conditional hosts; the associated feature backend/isolation options are reset through public recipe fields. Stationarity, KA2, optimizer reopening, DV12 latent perturbation and selected serving weights remain active when Atlas is selected. The resolved recipe, seed, initialization and external execution cap travel with the fixture state.

Image data retains the original ordered float32 template pixels and independent .01 Gaussian pixel noise clipped to [0,1]. Lossless compressed constants are target definitions, not checkpoints or generated samples; no archive path is required at runtime. The default budget remains 600 updates, or 480 for the original tiny/mean-only/uniform diagnostics.

Image quality requires the original RMSE/HQ/support bounds plus at most .10 nearest-partition TV and .10 TV including rejected mass. This is finite template fidelity, not pixel-law TV, OCR, topology, DSP, recognition or unseen-image generalization. Primary evaluation uses the actual public `ServedModel.sample/generate` law with a fixed isolated stream and `output_noise=False`. Atlas still perturbs latents and selects live/averaged weights; these results are distinct from historical clean atom enumeration.

## Queries that were previously missing their conditions

- **PR61 paired sparse occupancy completion:** the original top-half observed sensor map and mask enter G/D; each source identifies its matching bottom-half completion. Swapping the two generated completions preserves their marginal distribution but fails the paired gate. The claim is finite sparse completion, not general domain translation.
- **PR63 ambiguous masked completion:** a border-only input admits both original completions equally; two further contexts expose one distinctive pixel and select the corresponding completion. Uniform context weights preserve the original 50/50 output marginal. Each context must pass quality, support and mass gates, and observed-pixel RMSE must be at most .05. No observed pixels are hard-copied by the network.
- **PR65 finite RGB assignment:** the two original grayscale left/right shade inputs select specified red and blue RGB outputs. Paired quality and signed red/blue order reject grayscale output, swapped inputs and swapped channels. This claims only the two specified assignments, not natural-image colorization.

Each query retains separate transpose12 and residual16 cores. The conditional input projection/critic channel expansion and the RGB output expansion are visible architecture changes. The old unconditional variants remain accurately named and their historical outcomes are preserved.

## Five-word inverse and generation question

`image-five-words-joint-ae` retains `apple`, `grape`, `lemon`, `melon`, `berry`, the 28-character vocabulary, six positions including underscore padding, the original G/E/joint-D widths, and five learned 2D particle rows. Its default is public KA2 with the source's effective LR, spread, noise and EMA settings; Atlas is an explicit adaptation with conditional row controls disabled. The public generator optimizer uses homogeneous G/E/table groups for policy ownership. The adversarial joint objective preserves the original inverse-learning question; no fixed lookup substitutes for reconstruction.

The execution budget is **20,001 updates**, reflecting the original inclusive loop; the source's recipe schedule horizon is **20,000**. The frozen word evaluator requires at least 100 draws, .95 quality fraction, all five words, accepted-word/rejection mass TV at most .10, exact correctly paired reconstruction and token confidence at least .90. Full six-token probabilities are checked, including padding. Argmax strings alone cannot pass. Text views display actual decoded generated/reconstructed rows, and separate onehot strips display their actual confidence.

## Retained controls and exact aliases

The spatially uniform generator remains an expected representation failure. The mean-only critic remains an information-negative control for equal-mass patch positions. Width2/z1 remains capacity stress. Failing their output-quality bound does not mean the public API is broken.

Three cases share an exact default execution definition: reserved residual bars4 with frozen bars4; PR58 residual16 with shipped intensity residual16; source intensity transpose12 with PR58 transpose12. `alias_of` links these declarations. No second independent capture is claimed. Recipe, seed, budget, serving, source or runtime changes require a separate execution; this provider never automatically skips or fabricates a run.

## Validation and scientific status

The focused software suite performs actual one-update checks on every image fixture, checks the word host on KA2 and Atlas, compares architecture forwards with the original cores, and proves that observation preserves exact model/optimizer/policy/RNG state and the next update. Native diagnostic NaN sentinels are compared byte-for-byte; outputs and scalar scoring metrics must remain finite.

All 330 analytic controls behave as declared: 80 positive references and 250 negatives. Image controls discriminate mass imbalance despite perfect HQ/support, collapse and mean images. Conditional controls additionally discriminate swapped/shuffled correspondence, observed-pixel corruption and channel errors. Word controls discriminate diffuse correct argmax, wrong pairing, collapse, mass imbalance and wrong padding.

**No full-budget scientific run is claimed by this report.** The common runner adds actual training GIFs and source-bound binary readouts. A bounded 16/32-update illustration is explicitly incomplete for its full default budget; software conformance and oracle PASS do not imply model convergence, scientific qualification, or a historical quality-rating upgrade.

```sh
python -m pytest -q tests/test_toy_api_images.py
python -m benchmarks.toy_audit.api_run --list
python -m benchmarks.toy_audit.api_run --case image-pr63-conditional-residual_upsample16 --steps 32 --output /tmp/conditional-image-api-run
python -m benchmarks.toy_audit.api_run --case image-five-words-joint-ae --steps 32 --output /tmp/word-api-run
```

[Declaration/source bindings](declaration-provenance.json), [scorer controls](scorer-controls.json) and [software receipt](software-verification.json) retain the exact revision boundary. Bulk run states and logs remain outside Git.

## Every retained image bank

| Ordered bank | Historical entries | Exact finite question |
| --- | --- | --- |
| C_vs_O2 | pr131 | Fixed open C versus closed O pixel templates; RMSE is not an independent topology oracle. |
| L_chirality2 | pr73 | Fixed mirrored L templates; not chirality generalization. |
| T_junction2 | pr68 | Fixed T-junction patterns; not occlusion reasoning. |
| b_d2 | pr76 | Fixed mirrored b/d glyph templates; not general OCR or a dedicated chirality score. |
| barcode_quiet_lr2 | pr166 | Two fixed barcode-like templates with opposite quiet-zone placement; not barcode validity or decoding. |
| bars4 | develop-img_bars4, develop-img_residual_bars4, develop-img_tiny_generator | Undercapacity stress: width2 and a one-dimensional latent may constrain representation and optimization; non-blocking. |
| bars8 | develop-img_bars8 | Denser support stress: eight bar positions may exceed the short budget; failure cannot disqualify a controller. |
| blobs4 | develop-img_blobs4, develop-img_mean_discriminator | Low-information architecture stress: D sees only image mean; equal-mass patch positions are indistinguishable, so failure is non-blocking. |
| braille_cell_lr2 | pr170 | Left-heavy versus right-heavy raised-dot templates; tactile glyph asymmetry, not Braille decoding. |
| chirp_up_down2 | pr154 | Two fixed rising/falling spectrogram-like traces; not audio synthesis or frequency generalization. |
| colorize_lr2 | pr65 | Fixed grayscale left/right intensity patterns; no color channels or conditional grayscale-to-color query. |
| diag_ramp2 | pr62 | Fixed diagonal intensity ramps; not arbitrary image-algebra operations. |
| dof_center_edge2 | pr71 | Fixed center/edge focus profiles; not depth estimation or optics reconstruction. |
| dots_count23 | pr74 | Fixed two-dot/three-dot templates; not counting arbitrary objects, positions or cardinalities. |
| fg_bg_invert2 | pr66 | Fixed foreground/background intensity inversions; not conditional image inversion. |
| finder_diag2 | pr79 | Two fixed diagonal finder layouts; not QR recognition or error correction. |
| hamburger_kebab2 | pr151 | Two fixed menu-icon layouts; not UI interaction or semantic object recognition. |
| intensity2 | develop-img_intensity2, pr58 | Shipped intensity2 data reused as an architecture counterexample; not an independent new problem. |
| letterbox_pillar2 | pr78 | Fixed letterbox versus pillarbox border placement; not aspect-ratio inference from arbitrary images. |
| mask_inpaint2 | pr63 | Fixed templates named mask-inpaint; no observed image or mask enters G, so no conditional inpainting is tested. |
| moire_beat2 | pr150 | Two fixed moiré/beat intensity patterns; not recovery of unseen frequencies or phase. |
| play_pause2 | pr106 | Fixed play/pause icon templates; not video dynamics or button behavior. |
| pyramid_valley2 | pr77 | Fixed bright-center versus dark-center radial intensity patterns; not shape-from-shading. |
| radial_wedge2 | pr67 | Fixed radial/wedge patterns; not a segmentation or reconstruction task. |
| ramp_corner2 | pr70 | Fixed corner-ramp intensity templates; not a learned coordinate system. |
| smile_frown2 | pr69 | Fixed smile/frown arcs; not emotion classification or facial-image fidelity. |
| soft_ring2 | pr59 | Fixed soft radial/ring templates; not a continuous stochastic shape family. |
| sonar_echo_near_far2 | pr159 | Two fixed near/far echo-location templates; not acoustic propagation or range inference. |
| sparse_obs2 | pr61 | Fixed sparse-observation-like templates; no source-domain input or paired correspondence tests translation. |
| stairs_asc_desc2 | pr75 | Fixed ascending/descending stair templates; not sequence reasoning. |
| stripes2 | develop-img_stripes2, develop-img_uniform_generator | Known representation failure: G can only output spatially uniform images, so stripe quality is impossible; diagnostic only. |
| swirl_cw2 | pr72 | Fixed opposite-handed swirl patterns; not rotation dynamics or optical flow. |
| traffic_stack_rg2 | pr80 | Two fixed grayscale traffic-stack patterns; not red/green color semantics or traffic rules. |
| vh_bars2 | pr64 | Fixed vertical/horizontal bars; same orientation-coverage question as the shipped stripes family. |
