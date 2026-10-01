# Original qualification for RA16-portability

Evidence: **VALID**. Original quality qualification: **PASS**.

| Scope | Task | Quality | Executed source |
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

15 original RA14 quality executions,4 fresh RA15 affected gates, RA13 fresh learned training; latest source qualifies through explicit bridges and actual latest CUDA replay/full suite.

Qualification base: PR155 f459cb6d6aaaabeb1af076ec53ad7a963618de90.
The observed PR155 upstream head cabe2084284db923d525918cbf3e18de6f20faac requires its separate source/execution bridge and suite; this closure does not qualify that source.
Moving quality uses COMPLETION.verdict.periods and the original scorers. Native gates retain34 observations, five20k terminal draws and the100k holdout.
RA14 moving rotated100 qualityFAIL remains in history. Earlier adapter errors, NaN comparison failure, cancelled RA15 suite and unlaunched RA15 replay remain retained.
Toy original gate: PASS; all9 postupdate metric/LR records match RA11. MNIST all10 metric/LR records match corrected E22; no numerical MNIST gate was added.
Inherited final learned metrics: {"mnist": {"active_embedding": {"embedding_frechet": 0.5444880363760234, "embedding_precision": 0.869140625, "embedding_recall": 0.84716796875}, "class_mass_tv": 0.03639648437499999, "confident_class_coverage": 10, "confident_class_mass": [0.082275390625, 0.09033203125, 0.064453125, 0.065185546875, 0.079345703125, 0.066162109375, 0.074462890625, 0.070556640625, 0.05908203125, 0.07568359375], "confident_fraction": 0.7275390625, "mean_classifier_confidence": 0.9037254452705383, "pixel_clipping_fraction": 0.40906819701194763, "predicted_class_mass": [0.096923828125, 0.099853515625, 0.0869140625, 0.1015625, 0.109130859375, 0.08984375, 0.09814453125, 0.097412109375, 0.102783203125, 0.117431640625], "raw_embedding": {"embedding_frechet": 6.852170897247333, "embedding_precision": 0.8798828125, "embedding_recall": 0.85546875}}, "toy": {"centre_distance": 0.042550504207611084, "clean_particle_centres": {"centre_distance": 0.02287120744585991, "coverage": 25, "covered_centroid_rms": 0.010440723970532417, "mass_tv": 0.0366796875, "precision": 0.9990234375, "supported_mass": [0.037109375, 0.041015625, 0.03515625, 0.0341796875, 0.0390625, 0.03515625, 0.041015625, 0.04296875, 0.0361328125, 0.041015625, 0.0400390625, 0.0390625, 0.037109375, 0.048828125, 0.0380859375, 0.0361328125, 0.041015625, 0.044921875, 0.041015625, 0.0361328125, 0.041015625, 0.0458984375, 0.04296875, 0.04296875, 0.041015625], "unsupported_mass": 0.0009765625}, "coverage": 25, "covered_centroid_rms": 0.009175768122076988, "mass_tv": 0.05211425781250001, "precision": 0.96533203125, "supported_mass": [0.03515625, 0.037353515625, 0.0341796875, 0.0357666015625, 0.03857421875, 0.03271484375, 0.04248046875, 0.0391845703125, 0.036865234375, 0.0382080078125, 0.0399169921875, 0.0384521484375, 0.0350341796875, 0.0455322265625, 0.0390625, 0.033203125, 0.03857421875, 0.0421142578125, 0.037841796875, 0.03955078125, 0.03857421875, 0.045166015625, 0.0413818359375, 0.040771484375, 0.0396728515625], "unsupported_mass": 0.03466796875}}.
Latest original learned CUDA replay:40 actual updates, two10 branches for each fixture; fresh learned training remains2000 RA13 updates per fixture.
Actual latest full-suite counts: {"errors": 0, "failed": 0, "passed": 1445, "skipped": 12, "subtests_passed": 18, "xfailed": 0, "xpassed": 0}.
Actual pytest summary: 1445 passed, 12 skipped, 18 subtests passed in 393.52s (0:06:33).

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

Raw checkpoints, datasets, clouds and images stay local. This helper performed no numerical work.
