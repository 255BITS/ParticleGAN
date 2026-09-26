# RP2 frozen 22-task qualification route map

Existing runners cannot qualify RP2 unchanged. `public_default_verification.py` is a historical GAN-v3 **19-task** runner. Its GANTrainer routes receive `gan_v3_recipe`/`LegacyRecipe`; the legacy critic factory explicitly constructs K3P. Native toy100 and compatibility runners make the same historical recipe conversion, impose numeric horizons, and apply benchmark cosine/noise schedules.

## Frozen declarations

- 19 full task specs: `benchmarks/transfer_suite/plans/default_comparison.json`; discriminator cards: `reports/transfer_suite/unadjusted/leading_profile.json`.
- Native names: `benchmarks/toy100/problems.py`: grid100, rotated100, staggered100. Existing frozen resource declaration: `configs/toy100/constraints_simple_regularization.json` (7000 updates, training seed 1234, affine_square_v1, 20,000 particles, latent dimension 2, batch size 2,048). Preserve these resources/initialization while replacing the learner settings explicitly.
- Quality rules: `benchmarks/transfer_suite/protocol.py::test_verdict`, `locked_shared/observation.py::sustained`; native `toy100/gate.py` **and** `accuracy_gate.py`. Twenty-four complete transfer observations and native coverage plus five final accuracy checks and fixed holdout remain mandatory.

## Route map

| Tasks | Count | Faithful route |
|---|---:|---|
| vector_two_broad, vector_unequal_mass, vector_unequal_width, vector_anisotropic, vector_overlap, vector_spiral |6| Current GANTrainer after runner adaptation |
| img_stripes2, img_bars4, img_blobs4, img_intensity2 |4| Current GANTrainer after runner adaptation |
| grid100, rotated100, staggered100 |3| Current GANTrainer after native runner adaptation |
| mode_hold |1| Can migrate to current GANTrainer while preserving its frozen host fixture |
| two_pole |1| Direct-particle custom loop; no ordinary generator |
| trajectory, residual_student |2| Conditional pairing plus auxiliary objectives |
| unipolar, mid_scale_identity |2| Multiple scale-conditioned roles per update |
| ae_gan_hold |1| Encoder/routed prior/reconstruction and hold objectives |
| cover_leftover |1| Residual generator, two priors, auxiliary objectives |
| unused_token_hold |1| Slot embeddings and unused-control hold objective |

Thus **14 routes are compatible with GANTrainer in principle; eight need a component-controller binding**. Unmodified current GANTrainer cannot execute all 22 faithfully.

## Runner adaptations

Use current `get_recipe(total_steps=None, continuous_precision="rp1", adam_eager_state=True)` and change only each frozen host's declared resources. Keep evaluator budgets outside the learner. Remove historical cosine hooks and externally scheduled noise wrappers; the package must own its exact fixed 360/720 bootstrap once. Existing `host_recipe`, `resolve_config`, `rate_action` and `step_with_policy` are not directly compatible.

Keep full current recipe identity in receipts. Legacy serialization and `GLOBAL_RECIPE_FIELDS` omit precision/eager flags. Add a separate continuous receipt schema with complete applied-rate/noise/controller traces; retain quality scorers and thresholds unchanged. Snapshot package plus all adapters. Installed-wheel origin/hash checks are reusable.

## Learner API gaps

Recipe optimizer factories construct eager KA2 components but **do not instantiate or advance RP2 precision**. That happens only in GANTrainer. A factory-only custom loop would silently train another learner. The eight custom hosts need the same package controller bound to paired updates, real-data gradient disagreement, normalized update activity and checkpoints. Conditional views must differentiate data coordinates consistently; multiple priors need explicit A2 ownership; direct particles need the public direct-response path. Preserve host losses and conditioning rather than forcing a dummy unconditional architecture.

Reusable scaffolding: `public_default_verification.py` host/model construction and `public_module_manifest`; `vector_tasks.py` and `image_tasks.py` samplers/scorers; `locked_shared` host files; `toy100/train.py` native model/initialization/evidence scaffold; `suite.py::snapshot/verify_source`; `toy_suite.py` aggregation. Historical schedule/identity checks need a versioned continuous route, not disabled grading.

Keep all unimplemented routes NOT_RUN. An installed-wheel GAN-v3 control, six-case screen, research hooks, endpoint, EMA result or ancestor 22-task result cannot supply missing RP2 passes. No training, tests or worker changes were performed.
