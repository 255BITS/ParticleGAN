# C6 remaining 13 trainer routes

After the image pass, the remaining trainer-compatible tasks are **3 images, 6 vectors, 3 native100 tasks and mode_hold**. Use the identical C6 package and `get_recipe(continuous=True, critic_memory="moving", game_update="secant_resolvent", lr_floor=1., network_lr_floor=1.)`, changing only frozen resource dimensions. Every trainer needs `serial_backward=True`, nonfused/nonforeach Adam. Evaluator budgets stay outside learner policy.

| Group | Exact tasks | Reusable scaffold and necessary adaptation |
|---|---|---|
| Images | img_stripes2, img_bars4, img_blobs4 | Verified `experiments/constant_game/image_gate.py` already accepts these names and reads the full frozen plan. Preserve all24 checks/five-pass suffix. |
| Vectors | vector_two_broad, vector_unequal_mass, vector_unequal_width, vector_anisotropic, vector_overlap, vector_spiral | `public_default_verification.py::setup_vector` host scaffold; `vector_tasks.py` sampler/scoring. Replace historical recipe construction only. Preserve prior-first CPU initialization with std.5/seed0, data0/latent1/penalty2 streams, and promoted discriminator cards in `leading_profile.json`. Budgets1200 exceptspiral1600;4096 evaluation samples, latent seed990, isolated paired output-noise namespace402+1901; all24 observations and five suffix. |
| Native100 | grid100, rotated100, staggered100 | `toy100/train.py` initialization/data/evidence scaffold plus unchanged coverage/accuracy/holdout scorers. Seed1234,7000updates,20k/z2/batch2048. Actual generator is identity-initialized Linear2→2 under affine_square_v1; prior is redrawn uniform[-5,5]; critic width128/layers3/Fourier3. Preserve exact native RNG order. Remove wrapper noise and `step_with_policy`; package owns C6 updates/noise. Keep both coverage and accuracy gates, last-five checks and fixed100k independent holdout. |
| Tiny ring | mode_hold | `locked_shared/mode_hold.py` fixture:12particles,z4,batch128,width96×3/Fourier3,prior std.5,seed0,1200updates. Requires special attention to shared data/latent RNG order. |

**mode_hold is not a drop-in callable port.** Its frozen stream order is D-real → D-latent → G-latent → G-real. C6's transaction materializes `generator_real()` before checkpointing, which moves G-real ahead of both latent draws if the callable shares their stream. Preserve the original draws through an explicit deterministic harness adapter or package binding before declaring exact fixture parity. Vector/native real-data streams are independent of their latent streams, so this particular reordering does not change their sampled values.

Do not use `LegacyRecipe`, `gan_v3_recipe`, old `declared_recipe` serialization, external LR/noise schedules, or RP2 flags. Factories alone omit C6's two-field joint update. Old native identity/schedule receipts need an explicit continuous-C6 schema while all numerical quality criteria remain intact.

No new learner mechanism is required for these13 unconditional scalar hosts. The eight custom formulations still require a faithful joint-game component binding, including their conditioning/auxiliary losses and transactional state. Keep every unrun route NOT_RUN. Full frozen task specs, discriminator cards and source hashes are in the paired JSON. No implementation, training, tests or worker edits were performed.

Subsequent supervisor update: C6 stationary failed30 checks at3390–3680 with minimumHQ0, then recovered. No further broader C6 execution is authorized. The image and checkpoint passes retain their own scope; this route map is preparation evidence for successors.
