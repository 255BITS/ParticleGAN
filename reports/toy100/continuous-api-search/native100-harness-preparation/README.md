# Three native100 host routes to the public trainer

Preparation only; all three tasks remain NOT_RUN here. No Torch import, installation, model/test/training execution, GPU work, or active-checkout edit occurred. `source-map.json` records exact host hashes, definition locations, fixture fields, seed namespaces and candidate source checks. Host root: `/ml2/hypergan/ParticleGAN-k3p-continuous-search`. Candidate reference is immutable DV6 long-full `source.zip`, SHA256 `b9a6a43dc53862ac9c0823e1f3e3fe9ca970edeac53d0e2e9ba741336bef3ce4`; all 30 members match the declaration, and the inspected package matches that ZIP. This is not a DV6 quality verdict.

The tasks are `grid100`, `rotated100`, and `staggered100`. They share the same host and learner. `problems.py::sample_real` draws a uniform mode index then isotropic Gaussian noise of std .03. Grid coordinates are -4.5 through 4.5; rotated100 rotates the grid 25 degrees; staggered100 alternates second-coordinate shifts -.25/+.25 and scales the first coordinate .85. Only unlabelled samples enter training; geometry stays inside the sampler/evaluator.

## Exact initialization and public construction

Read `constraints_simple_regularization.json` together with `train.py::make_trainer` and `_init_linear`, not the JSON alone. The host uses seed1234, 20,000 learnable prior rows, z_dim2, batch2048 and `affine_square_v1`. G is a trainable Linear(2,2), identity weight and zero bias (six parameters); g_hidden128 is inactive. D is `lib.toy_models.SimpleMLPDiscriminator(2,128,3,fourier=3)`: three width128 LeakyReLU(.2) layers, linear scalar head, and pi/2pi/4pi sine/cosine features. D weights are redrawn Xavier-uniform, biases zero. G remains trainable, so preserve normal latent-table prior optimizer semantics rather than treating the prior as direct output particles.

The exact RNG order inside a fork is: seed CPU and selected CUDA backend1234; construct ordinary ParticlePrior on CPU (normal std1 draw, even though later discarded); move prior to selected device; redraw its coordinates uniform[-5,5] on that device; construct Linear defaults on CPU, move it, overwrite weight with identity and bias zero; construct D defaults on CPU, move D; redraw each D Linear weight with Xavier-uniform on its current device, then zero biases. The discarded constructor draws are part of the fixture. Do not substitute prior std.5, skip discarded draws, move uniform initialization earlier, or replace D with the locked-ring width96 model.

Reuse DV6's exact package construction, changing only native host resources/seed/models:

```python
recipe = get_recipe(total_steps=None, continuous_policy="dv6",
                    input_noise_std=0., output_noise_warmup=0.,
                    num_particles=20000, z_dim=2, batch_size=2048)
trainer = GANTrainer(recipe, G, D, prior=prior, seed=1234,
                     serial_backward=True,
                     optimizer_options={"foreach": False, "fused": False})
trainer.step(real_d, generator_real=lambda: sample_real(
    problem, 2048, device=device, generator=data_rng))
```

G/D base rates .00425 and prior base .0085 come from the unchanged candidate recipe; DV6 itself owns adaptive applied rates/controller behavior. Do not add an external scheduler or clamp its rates. Seven thousand updates is the evaluator budget only, never a learner total_steps. Do not use LegacyRecipe, old resolve_config, noise wrappers, `step_with_policy`, or a factory-only update. The public trainer already accepts the exact prior, affine G, scalar D and second real batch; no training mechanism change is needed.

Data uses its own device generator seed1234, with one fresh D-real draw and another fresh G-real draw per accepted update. Trainer latent seed is1236; reserved penalty seed1237; default eval seed1238; DV6 private training-noise seed1239. Data and latent streams are independent, so `generator_real` is compatible. Save the data generator alongside the complete trainer checkpoint. DV6 owns zero input noise and constant output std.029, including step0. This intentionally replaces the old learner's 700/1400-update noise ramps, without changing evaluator random-number namespaces.

## Observation and noise adapter

There are **34 observations per live/EMA model**: 0,1,10,25,50,100, then every250 through7000. Each uses20,000 sampled prior indices with replacement, not prior enumeration. A fixed20,000 target cloud uses seed1234+401=1635. Every observation resets latent seed1234+403=1637. Snapshot arrays are the first4096 of those same scored live/EMA/target clouds.

The frozen config does **not** set `output_noise_rng="isolated"`. Its ordinary OutputNoise draws from a global RNG fork seeded1234+402=1636. `GANTrainer.sample`'s nested fork restores that seed separately for live/EMA, yielding paired output noise. This seed stays fixed across observation steps. Neither +1901 nor a step offset belongs to this native protocol.

Current DV6 `GANTrainer.sample(..., generator=...)` uses the same supplied generator for latent IDs and output noise. Therefore directly calling it with1637 changes the frozen output-noise draws. It also has no separate evaluation-noise-stream argument. The required adapter is evaluation-only: preserve all model modes and training RNGs; in a fork, set model/prior to eval; obtain latent draws from the appropriate live/EMA public prior with a fresh1637 stream; evaluate the corresponding public G; add candidate-owned output sigma times `torch.randn_like(output)` under the independent1636 global seed; restore modes/RNGs. Repeat with the same seeds for live and EMA. Candidate `output_noise_std(recipe, completed_steps)` supplies the amplitude. An optional public sample noise-stream argument could replace this adapter, but is not required to train and must not be introduced as an unreviewed mechanism change. Retain receipts that observation does not advance any training stream.

The independent final holdout uses100,000 samples with offsets target1601, noise1602, latent1603: actual seeds2835/2836/2837. Use the same separate-noise adapter for each live/EMA holdout, with noise paired and training state untouched. `AccuracyEvidence.finish` currently calls trainer.sample directly, so it needs an injected sampling adapter or an equivalent narrow evidence-host adapter; reusing it unchanged would silently change the noise namespace.

## Frozen verdict and evidence

Keep the original numeric functions in `metrics.py`, `accuracy.py`, `gate.py::score_run`, and `accuracy_gate.py::score_run`. Coverage requires100 modes, at least.005 HQ mass in every mode (100 of20,000), precision>=.97 within radius.09, total mass TV<=.10, largest mode<=.02, per-mode covariance eigenvalue ratios .40–1.70 and radial median ratios .65–1.40, with all values finite. It requires at least five consecutive passing terminal observations; step0 cannot establish convergence. The final five are6000,6250,6500,6750,7000.

Accuracy additionally requires mass TV<=.06, center RMS/std<=.20, absolute conditional covariance trace bias<=.10, and radial KS<=.04 at each of those five exact20,000-draw clouds **and** the separate100,000 holdout; the frozen coverage rule also applies. Live is primary; EMA is diagnostic. Saved target clouds must pass the oracle audits. Do not substitute an average, best checkpoint, or ordinary mode-count-only gate.

Preserve all34 live/EMA events, monotonic elapsed times,34 snapshot NPZs, exact final_samples.npz, five quality_checks NPZs, holdout_samples.npz, hashes and recomputed verdicts. The gate compares complete schedules and rescored saved clouds, not stored `passed` fields. Existing legacy manifest resolution rejects DV6's continuous_policy/total_steps=None and mixes learner schedule fields into host config. A continuous-public receipt/manifest adapter must validate the frozen host fields separately from the actual candidate recipe and then invoke unchanged numeric scoring. Do not fake a legacy recipe or remove provenance validation to make the old wrapper accept it.

## Backend decision still required

The frozen constraints JSON names CPU; native code also explicitly supports a declared CUDA device override. That override affects prior uniform draws, D Xavier draws, data, latent and evaluation streams. Thus an all-CPU fixture moved to CUDA does not reproduce the native CUDA initialization, and same integer seeds do not make CPU/CUDA random samples identical. Before execution, identify the authoritative native host backend/fixture and preserve its order. If CUDA is selected as the host backend, record it and its initial tensor/stream hashes; do not claim CPU bit parity. If exact CPU sampling is required while training CUDA, the public trainer rejects a CPU latent generator for CUDA parameters, which is a real binding gap requiring a reviewed explicit adapter. No backend approximation was made here.

For a later reviewed runner, snapshot the candidate package, host/scoring dependencies and adapter; record full resolved recipe, fixture/backend, initialization hashes, runtime/determinism, all applied rates, controller/noise state, and external data RNG. No source implementation or training loop is provided by this preparation.
