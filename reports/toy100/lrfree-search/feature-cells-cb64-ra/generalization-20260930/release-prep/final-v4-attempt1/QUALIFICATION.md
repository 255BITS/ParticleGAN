# Current PR155 original qualification

Evidence: **VALID**. Original quality qualification: **PASS**.

Current PR155 base: cabe2084284db923d525918cbf3e18de6f20faac.
Frozen current package: 500ff0e966beb649dd7cafa0b91d7bb30cb451e5d62ece2883411a0507c8df61.

| Scope | Task | Quality | Actual executed source |
|---|---|---|---|
| portability | mode_hold | PASS | RA14-replay |
| portability | img_intensity2 | PASS | RA14-replay |
| portability | img_blobs4 | PASS | RA14-replay |
| portability | img_bars4 | PASS | RA14-replay |
| portability | vector_unequal_mass | PASS | RA14-replay |
| portability | img_stripes2 | PASS | RA14-replay |
| portability | vector_two_broad | PASS | RA14-replay |
| portability | vector_unequal_width | PASS | RA14-replay |
| portability | vector_anisotropic | PASS | RA14-replay |
| portability | vector_overlap | PASS | RA14-replay |
| portability | vector_spiral | PASS | RA14-replay |
| portability | stationary | PASS | RA14-replay |
| native | grid100 | PASS | RA14-replay |
| native | rotated100 | PASS | RA14-replay |
| native | staggered100 | PASS | RA14-replay |
| moving | grid100 | PASS | RA15-partial-recovery |
| moving | rotated100 | PASS | RA15-partial-recovery |
| moving | staggered100 | PASS | RA15-partial-recovery |
| portability | ring_shift | PASS | RA15-partial-recovery |

15 RA14 executions plus4 fresh RA15 affected gates; none relabelled freshRA17.
The current source bridge preserves original finite-horizon math through all21 prefixes: lifetime stationary decisions imply table s≥.25 at every step. Lazy diagnostics and valid routing preserve forward/gradient/RNG/schema law. No universal noise-floor training parity is claimed.
Moving quality remains actual scorer periods, both original turns over1500 updates. Native coverage retains34 observations, five20k terminal checks and100k independent holdouts.
Fresh learned training remains RA13,2000 updates per fixture. Toy original gate: PASS; all9 metric/LR records match RA11. MNIST all10 records match corrected E22; no numerical MNIST gate was added.
Inherited final learned metrics: {"mnist": {"active_embedding": {"embedding_frechet": 0.5444880363760234, "embedding_precision": 0.869140625, "embedding_recall": 0.84716796875}, "class_mass_tv": 0.03639648437499999, "confident_class_coverage": 10, "confident_class_mass": [0.082275390625, 0.09033203125, 0.064453125, 0.065185546875, 0.079345703125, 0.066162109375, 0.074462890625, 0.070556640625, 0.05908203125, 0.07568359375], "confident_fraction": 0.7275390625, "mean_classifier_confidence": 0.9037254452705383, "pixel_clipping_fraction": 0.40906819701194763, "predicted_class_mass": [0.096923828125, 0.099853515625, 0.0869140625, 0.1015625, 0.109130859375, 0.08984375, 0.09814453125, 0.097412109375, 0.102783203125, 0.117431640625], "raw_embedding": {"embedding_frechet": 6.852170897247333, "embedding_precision": 0.8798828125, "embedding_recall": 0.85546875}}, "toy": {"centre_distance": 0.042550504207611084, "clean_particle_centres": {"centre_distance": 0.02287120744585991, "coverage": 25, "covered_centroid_rms": 0.010440723970532417, "mass_tv": 0.0366796875, "precision": 0.9990234375, "supported_mass": [0.037109375, 0.041015625, 0.03515625, 0.0341796875, 0.0390625, 0.03515625, 0.041015625, 0.04296875, 0.0361328125, 0.041015625, 0.0400390625, 0.0390625, 0.037109375, 0.048828125, 0.0380859375, 0.0361328125, 0.041015625, 0.044921875, 0.041015625, 0.0361328125, 0.041015625, 0.0458984375, 0.04296875, 0.04296875, 0.041015625], "unsupported_mass": 0.0009765625}, "coverage": 25, "covered_centroid_rms": 0.009175768122076988, "mass_tv": 0.05211425781250001, "precision": 0.96533203125, "supported_mass": [0.03515625, 0.037353515625, 0.0341796875, 0.0357666015625, 0.03857421875, 0.03271484375, 0.04248046875, 0.0391845703125, 0.036865234375, 0.0382080078125, 0.0399169921875, 0.0384521484375, 0.0350341796875, 0.0455322265625, 0.0390625, 0.033203125, 0.03857421875, 0.0421142578125, 0.037841796875, 0.03955078125, 0.03857421875, 0.045166015625, 0.0413818359375, 0.040771484375, 0.0396728515625], "unsupported_mass": 0.03466796875}}.
Actual current learned replay:40 CUDA updates, strict native/CPU-map loss/state/sample parity and original native control agreement.
Actual current full suite: 1502 passed, 12 skipped, 18 subtests passed in 403.52s (0:06:43).
Required current CUDA nodes: [{"classname": "tests.test_e22_routed_validation", "name": "test_71_site_mix_has_no_scalar_readbacks_and_finish_has_exactly_one", "records": 1, "status": "PASS"}, {"classname": "tests.test_e22_routed_readbacks", "name": "test_cuda_scalar_reads_do_not_grow_with_routing_site_count", "records": 1, "status": "PASS"}, {"classname": "tests.test_e22_routed_readbacks", "name": "test_71_site_lazy_and_eager_observation_match_gradients_resume_and_serving[cuda]", "records": 1, "status": "PASS"}, {"classname": "tests.test_feature_portability", "name": "test_cuda_default_preserves_feature_reactions_and_checkpoint_replay", "records": 1, "status": "PASS"}].
Exact upstream CI CPU CLI:PASS, sites3/tokens2/particles16/steps2/warmup1. No timing quality threshold was added.

Actual skip reasons:

- SKIPPED [1] tests/test_gym_data.py:4: could not import 'gymnasium': No module named 'gymnasium'
- SKIPPED [1] tests/test_gym_lander_live.py:13: could not import 'gymnasium': No module named 'gymnasium'
- SKIPPED [1] tests/test_gym_probe.py:4: could not import 'gymnasium': No module named 'gymnasium'
- SKIPPED [1] tests/test_lunar_flight.py:6: could not import 'gymnasium': No module named 'gymnasium'
- SKIPPED [4] tests/test_cifar_resume.py:19: opt-in real-data CUDA integration test
- SKIPPED [1] tests/test_image_ddgan.py:72: could not import 'torch_fidelity': No module named 'torch_fidelity'
- SKIPPED [1] tests/test_particle_native_2d.py:29: platform-sensitive convergence gate; opt in with RUN_PARTICLE_NATIVE_RESEARCH_GATE=1
- SKIPPED [1] tests/test_toy100_device.py:45: asserts CUDA is absent
- SKIPPED [1] tests/test_toy100_device.py:55: asserts CUDA is absent

Historical failures, old-base closure, unlaunched lanes and earlier metadata failures remain retained. Weights, datasets and clouds stay local. The user-authorized shift visualization is tracked separately. This finalizer performed no numerical work.
