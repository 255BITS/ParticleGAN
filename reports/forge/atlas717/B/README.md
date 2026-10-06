# Atlas717 Track B recorded Tier1 results

This isolated six-row readout carries the original collector grades. It adds no training, sampling or regrading.
Gaussian, AE and ring already used latent sigma.025. Their type/ownership comparison does not show that noise was added.
The two direct-coordinate controls retain their original Parameter law. Word retains the original N5 paired host and remains blocked.
Track A declares sampled NoisyParticlePrior(.025); Track B retains the existing MoG prior. Their Sources, requests and physical attempts stay separate.
All gates use the original24 observations and five-check terminal suffix. A final scalar PASS alone does not establish stable PASS.
This report awards no original-parent, main-inventory, family-default, winner or speed credit.

| Goal | Recorded status | Final scalar checks | Passing suffix | Saved observations |
| --- | --- | --- | --- | --- |
| Acquire Gaussian location, width and CDF shape. | FAIL | sample_count=4096 (>= 4096: PASS); finite_fraction=1.0 (== 1.0: PASS); mean_error_sigma=0.010363838053308427 (<= 0.2: PASS); std_ratio=0.9947513128238696 (>= 0.8: PASS); std_ratio=0.9947513128238696 (<= 1.2: PASS); cdf_ks=0.03311356661847087 (<= 0.05: PASS) | 1 | [GIF](media/gaussian1d_acquisition.gif) |
| Move live coordinates while keeping critic gradients bounded; pole balance is ungated. | FAIL | mean_abs=0.002445647493004799 (>= 0.3: FAIL); grad_med=0.010896995663642883 (<= 1.0: PASS) | 0 | [GIF](media/two_pole.gif) |
| Move the concept while holding unused controls. | BLOCKED | No accepted scalar gate cells | Unavailable | No accepted media |
| Preserve paired reconstruction and generated-anchor hold. | PASS | recon_mse=0.004397972021251917 (<= 0.05: PASS); hold=0.003759577637538314 (<= 0.35: PASS) | 20 | [GIF](media/ae_gan_hold.gif) |
| Acquire all16 modes with balanced mass, precision and noncollapsed spread. | FAIL | sample_count=4096 (>= 4096: PASS); modes=16 (>= 16: PASS); mass_tv=0.0625 (<= 0.15: PASS); hq=0.88134765625 (>= 0.85: PASS); component_covariance_error=4.24218525364995 (<= 0.85: FAIL); component_min_eigen_ratio=0.3165477216243744 (>= 0.15: PASS) | 0 | [GIF](media/ring16_acquisition.gif) |
| Acquire five paired words and confident exact reconstruction, including padding. | BLOCKED | No accepted scalar gate cells | Unavailable | No accepted media |

Each row’s original thresholds, scalar24-point curve, source/request/task/attempt hashes and saved-media proof are in [RESULTS.json](RESULTS.json).
The relative source-file hash index is [SOURCE.json](SOURCE.json). Original requests, terminals, arrays, states, logs and private media receipts remain local.
These unique physical attempts used 411.572834875s; the full declared ceiling was2220s. Reporting metadata and previous campaigns are excluded from this local total.
Reproduction requires the declared byte-hash index and matching task/request. The source origin commit alone does not guarantee that every frozen input is available; plain develop and another track are not substitutes.
