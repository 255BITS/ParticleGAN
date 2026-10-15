# Delayed quarter-G continuation: rejected at the original horizon

The exact retained treatment completed native updates **1051–1200 once**, then stopped. The original scorer returns **FAIL: terminal passing suffix 0/5**. The retained native control remains **FAIL: suffix 3/5**. Quarter-G prevented the local1050 dip, but did not preserve terminal quality.

| Update | Retained control HQ | Delayed quarter-G HQ | Treated HQ modes | Treated instantaneous gate |
|---|---:|---:|---:|---|
| 1000 | 0.926758 | 0.926758 | 8 | PASS |
| 1050 | 0.869385 | 0.956055 | 8 | PASS |
| 1100 | 0.939453 | 0.900879 | 8 | PASS |
| 1150 | 0.933105 | 0.912598 | 8 | PASS |
| 1200 | 0.989258 | 0.833008 | 8 | FAIL |

The original gates remain modes≥8 and HQ≥0.9 at the original24 scheduled clean-LIVE observations. All five terminal observations retain eight HQ modes. The complete treated curve uses the unchanged native prefix through1000, the previously retained treated1050 observation and the three new treated observations. No later native-control point fills the treated curve. There are10 passing observations, first at700, but the endpoint fails. This is a delayed intervention **beginning at1001**, never a from-start recipe, holdout or qualification result.

The final scheduled4096-row cloud has equal-cell center RMS **1.574047→2.397874 target σ**, full per-cell covariance Frobenius RMS **1.151093→24.804588**, and >6σ rows **0→358** (control→treatment). At1100/1150, treated covariance RMS is3.318968/29.112659 and >6σ rows129/329 despite passing HQ. Thus the original HQ gates can tolerate these descriptive tails; these metrics do not replace or tighten the frozen gate. Endpoint mode0 accounts for the largest covariance error, with trace71.259813 target-variance units and center error5.003547σ. Aggregate allocation is unchanged in every paired terminal cloud: zero nearest-mode assignment flips, and identical massTV, ending0.208496. No independent rare-tail claim is made from4096 rows.

The delta remains the exact external adapter: multiply only generator-role actual native optimizer lr by0.25 at joint optimizer entry (.012→.003), invoke unchanged native DualNorm/row-normalized optimizer and restore original group rates immediately. D rate.018 and particle rate.03, original NS/symmetric-BCAP1 objective, models,12-mean unstandardized MoG σ.025,128-row real batch, original1200 horizon and clean output0 stay fixed. Learned D/table states evolve along the treated trajectory. Native checkpoint metadata still stores original recipe/rates; the checkpoint-bound caller envelope is mandatory to identify this treatment.

The pinned d378 source and Torch2.13.0+cu126 runtime are reused from the earlier matched fresh cohort; this is not a byte-original Torch2.14 archive replay. **465/465 checks pass**: strict full1050 restore, every update's generator-only rate ownership/restoration and finite scalar losses, finite saved model tensors and named/native/global CPU/CUDA RNG equality to the retained control at1100 and1200. Only changed scripts were AST checked; unchanged source/runtime identities were cached. Mandatory normal Forge source snapshot and admission checks still ran. Native source was not edited.

One normal shared physical-GPU0 admission completed150 D and150 joint G/P optimizer calls, native300 G/900 D forwards,38400/115200 native fake/input rows and19200 regenerated original target rows. Only the original3 scheduled public clean draws were added,12288 rows; **zero diagnostic model forwards, control updates, prefix replay or holdouts**. Three before/after scheduled-boundary complete contexts and actual consumed D-real/G-fake/native-gradient packets are retained. TRAINING.gif shows those actual native128-row fakes, with its first frame visually inspected.

New paid wall **25.582731559s**, conservative charge **30.582731559s**, own cumulative **2887.108602629/4000s**. Measured native synchronized training18.227582s; setup/import/restore/input load2.796711s; target preparation0.080947s; scheduled sampling0.004152s; scoring/shape/checkpoint0.128726s. These subsets are already in paid wall. Preparation copy/input-binding0.184552s is reported separately; source-admission/reporting timing is unknown. Prior costs/errors remain unchanged.

**Decision: reject sustained constant quarter-G after1000 on this profile.** The local1050 prevention remains valid but does not generalize over the remaining original horizon. The parent requested a smallest generality control only if the suffix passed; that condition did not occur. No from-start control, adaptive trigger, additional factor, extension or integration is launched or recommended by this receipt. Healthy Ring16 PASS26, historical RA14 noisy wins and the other owners' case-specific evidence are unchanged.

Evidence: [raw result](admission/6d71fc4827137838/results/RESULT.json), [original scorer grade](admission/6d71fc4827137838/results/GRADE.json), [complete treated curve](admission/6d71fc4827137838/results/CURVE.json), [caller envelope](admission/6d71fc4827137838/results/CALLER_TREATMENT_ENVELOPE.json), [protocol](PROTOCOL.json), [cost](COST.json), [actual-training GIF](admission/6d71fc4827137838/results/TRAINING.gif).
