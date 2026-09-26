# Independent preflight: public K3P vector reference

No source-level blocker remains for the authorized external worker to execute this one supporting `vector_unequal_mass` baseline. This is a source preflight, not a runtime or quality pass. No Torch import, model construction, training, backward, optimizer step or GPU computation ran during this review.

The reviewed bundle is under `/ml2/hypergan/ParticleGAN-k3p-continuous-search/reports/toy100/continuous-api-search/supervisor-support/k3p-vector-reference`. Its final `bundle-sha256.json` hash is `4495314edb77e5ac37e612169324d4d7d05e8df73df760fe2213aaf12dade417`. This audit lives outside that sealed bundle. `preflight-audit.json` pins reviewed files and records the checks and remaining runtime limits.

## Release and host identity

All 12 package files match the release ZIP, declared manifest, and actual git objects at v0.8.0 commit `0ff9a7afe5dcb828239369446cfe71971bce687b`. No package patch is present. The worker rejects an already-imported ParticleGAN, puts the isolated source directory first, and checks imported package paths/hashes. Original host sources remain provenance files and are not imported as a research package. In particular, `source/host_device.py` is retained provenance only; its broad process-wide device policy is not installed.

All 16 extracted model/scoring/protocol definitions match complete original source spans byte-for-byte. The unequal-mass spec and promoted `batchfeat_center6_distance_head` card match the pinned plan/profile. Its public BatchDistance constructor arguments match the canonical host branch. All 15 raw fixture tensors—7 G/prior and 8 D—match their declared shapes and hashes, independently checked without loading Torch tensors. Before training, the worker additionally requires full G/D/prior state hashes to match the retained RP5 initialization receipt, including buffers.

## Device, RNG and native optimizer contract

Prior/G/D construction occurs on CPU with global seed0 and the prior's separate CPU seed0, then the complete canonical parameter fixture is copied before moving models to CUDA. Data/latent/penalty streams use explicit CUDA seeds0/1/2; released trainer noise uses seed5. The public trainer draws its fresh generator-real batch through the callback once in its normal update order.

`host_cuda` temporarily routes implicit tensor allocation and implicit Generator construction to CUDA only while running the frozen sampler/scorer. It restores both the device context and original Generator class on success or exception. The worker asserts CPU default allocation and the original Generator class before trainer construction and before/after each update. During the generator-real callback the CUDA scope closes before native G/prior optimizer work resumes. The temporary Generator subclass uses the same native-constructor pattern as the original host device adapter.

The worker uses released `GANTrainer.step` and native lazy Adam. It requires initially empty optimizer state and does not create moments, relocate counters, install schedule hooks, patch optimizer steps, or replace released EMA arithmetic. `foreach=False`, `fused=False`, noncapturable Adam and native CPU scalar step counters with CUDA parameters/moments are verified after actual updates. Unexpected placement is retained as ERROR evidence without repair. The full public step, including nested higher-order graph construction and backward, is enclosed in the restoring serial-autograd context.

One small receipt issue was identified and resolved before sealing: the final optimizer device proof would have overwritten the first-update proof. The final worker writes separate `optimizer-device-proof-0001.json` and `optimizer-device-proof-1200.json`. This changes receipts only.

## Schedules and observation fidelity

The only recipe overrides are `total_steps=1200`, `num_particles=256`, `z_dim=4`, and `batch_size=128`. This is the released K3P schedule with an explicit benchmark horizon; it is not literal `get_recipe()`'s 7000-update configuration. Input noise reaches zero at completed120, output noise reaches .029 at240, and network/prior decay starts720. The final optimizer update1200 uses schedule index1199. The worker verifies every applied group rate against the released helper without modifying it externally.

Evaluation retains all24 steps50…1200 and4096 samples, the exact original scorer, live-primary/EMA-diagnostic split, and final-five sustained requirement. It uses latent990, target991, projections992 and paired isolated output-noise2303. The global402 seed is still set inside the forked measurement namespace but does not replace the explicit2303 output stream. This matches the declared current comparison protocol; historical global402 scores are not borrowed. CPU/CUDA RNG and model modes are restored, and a full trainer/data-state digest must remain unchanged around live and EMA observations.

The external checkpoint envelope binds source/fixture/protocol identity and serial execution mode before calling the unchanged release schema3 loader. The worker does not declare a resumed quality run; no continuation claim is inferred from the envelope check.

## Verification and handoff

Independent complete-file review and the standard-library preflight both passed against the final manifest. The latter checks hashes, exact release/host definitions, fixture storage, recipe/task/card/schedules, restoring scope behavior with a stub, full-step serial containment and envelope rejection. Stub checks do not certify native CUDA runtime behavior. The external worker's assertions still need to validate actual imports, model-buffer hashes and Adam placement during the already-authorized baseline execution.

Launch in the declared Torch2.13.0+cu126/CUDA12.6/A6000 environment and selected external lane:

```bash
/tmp/pr38-default-env/bin/python /ml2/hypergan/ParticleGAN-k3p-continuous-search/reports/toy100/continuous-api-search/supervisor-support/k3p-vector-reference/worker.py --output /absolute/new/reference-output
```

No shared evidence manifests or active candidate packages were edited by this review. A measured result will describe this finite scheduled reference only; it supplies neither a fourth mechanism nor continuous-learning eligibility.
