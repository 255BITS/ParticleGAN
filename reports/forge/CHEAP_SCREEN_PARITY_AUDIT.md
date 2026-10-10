# Cheap-screen parity audit — 2026-09-28

The historical K3P `two_pole` PASS is valid historical evidence, but does not qualify the current Forge formulation. The archived run and current public-component adapter differ in effective direct-particle LR, particle L2, RNG streams, backend/runtime, and—in the fresh-checkout negative control—the declared critic penalty. This audit establishes those differences; it does **not** identify which caused a metric change, propose threshold tuning, or establish a current-baseline false rejection.

Scope: source/receipt inspection and construction-only tensor/optimizer checks, followed by the separately authorized receipt-metadata correction below. **Zero training updates**, no seed experiments, and no optimizer, objective, or task-config changes. The inspected current source is snapshot `a565c608c1871a1de6e9c7d326246ac976eaa08a573cc0d697421a377a0958a4`, origin commit `5a30cd3db3577404b4747ba9a2fae398c2257a23`. Scientific comparisons below refer to that snapshot; the metadata correction creates a subsequent source cohort.

**Evidence and observed outcomes**

The frozen [22-problem summary](../toy100/gap-fill-20260925/qualification-summary.json) records K3P PASS on all 22 problems: 19 transfer tasks and three native tasks at the native canonical seed. That is not 22 repetitions of `two_pole`. Its [two-pole receipt](../toy100/gap-fill-20260925/results/k3p-toy-two_pole.json.gz) is pinned in baseline commit `92dc03195080376ba6b56d273e5c3b929e54ce61`; the [normalized historical card](records/history-k3p-84363f69cb3a.json) preserves its provenance. Native seed 1234 in the generic config header does not describe this transfer host: the historical wrapper explicitly uses seed 0.

| Exact evidence | mean_abs ≥ .30 | grad_med ≤ 1.0 | Sustained verdict | Reported cost |
| --- | ---: | ---: | --- | ---: |
| Frozen K3P two-pole | .6452074051 | .1471946090 | PASS; 24 checks, passing suffix 10; first pass step 50, confirmed step 64 | 3.491859448 s outer historical probe |
| Current `forge-no-critic-penalty`, attempt `fafc3da7cecf4b1780bbaedf66141092` | .4477476180 | 1.2546037436 | FAIL; 80 updates, 24 checks, three passing checks, suffix 0 | 7.371452854 s charged execution; 2.615649054 s adapter |

The fresh-checkout [request](attempts/fafc3da7cecf4b1780bbaedf66141092/request.json) declares only `reg_coeff=0` as its candidate change; its [result](attempts/fafc3da7cecf4b1780bbaedf66141092/result.json) records an actual completed scientific FAIL, with finite state and 80 optimizer updates for both roles. It passed both metrics at steps 54, 57, and 60, then exceeded the slope limit from step 64. This is an informative negative control, not a rerun of historical K3P. The two cost definitions and compute backends differ; they do not support a speedup claim.

Earlier public baseline [attempt b766e89d](attempts/b766e89d1a0843ba9af94f2ce94e4273/result.json) and anchor-ablation [attempt c0178716](attempts/c0178716dd3c4d5695b391919bf9ab81/result.json) failed travel (`.106520914` and `.104824930`). Their source digests are respectively `aee8c454…` and `e2b2e0dc…`, rather than `a565c608…`. They cannot be combined with the new negative control to estimate a same-source mechanism effect. The coordinator's subsequent same-source cheap batch and its limits are reported separately in [CURRENT_SMOKE_READOUT.md](CURRENT_SMOKE_READOUT.md); this audit did not execute that batch or use it to infer a cause for the historical difference.

**Effective protocol comparison**

| Property | Frozen K3P all-22 receipt/runner | Current public-component two-pole | Assessment |
| --- | --- | --- | --- |
| Host budget and support | 80 D + 80 particle updates; 12 scalar particles | Same | The generic 7,000-step/20,000-particle/2,048-batch config header is not the effective host budget. |
| Target and architecture | Balanced deterministic ±1 poles, fixed ±.05 spread; 1→32→1 SiLU critic | Same host loop and target | No target/architecture drift found. |
| Initialization | Stored critic tensors, zero particles, CPU fixture copied to CUDA | Stored critic tensors and zero particles on CPU | All five initial tensor byte hashes match the archived optimizer proof. The named network initializer is intentionally skipped for this fixture. |
| Adversarial objective | RpGAN logistic | Public RpGAN logistic | Same D/G scalar formulas and D-then-G ordering. |
| Particle auxiliary loss | `particle_l2=0`, explicitly supplied by the comparison wrapper | Host default `.02 * particles.square().mean()` | Confirmed objective difference; see policy distinction below. |
| D base LR / Adam betas | `.00425`, `(0,.999)` | Same | No D-rate mismatch found. |
| Direct-particle base LR | `.00425 * prior_lr_mult(2) = .0085` | `.00425`; direct group lacks explicit LR override | Confirmed adapter-policy difference. The public factory itself obeys its documented default. |
| LR horizon and shape | Host horizon 80; anneal start .6; network cap 1,600 resolves to 80; network floor .01, prior floor .05 | Same multipliers and effective horizon | At completed update 79, network multiplier `.012383560297262576`, prior multiplier `.052287254830706516` match exactly. |
| Direct-particle response | `(0,.9)` betas; gain `1 + max(cos(center(g), center(previous_g)), 0)` | Same public response enabled | Historical response receipt explicitly measures scheduled initial LR `.0085`; gain is separate from the base-LR discrepancy. |
| Critic regularization | K3P coefficient 1, kappa 1, blend floor .01; anchor decay .999 starts at first blend; 66 pure-A + 14 blended calls | Baseline public settings preserve these formulas/lifecycle; fresh negative control explicitly sets coefficient 0 | Intentional candidate ablation. Static source comparison does not certify an identical baseline trajectory. |
| A2 and spike guard | No latent table; guard minimum 200 exceeds budget 80 | Same host applicability | Neither can be credited as exercised training protection here; synthetic component probes are separate evidence. |
| Noise amplitudes/schedules | Input .5 anneals over .1×80; output .029 warms over .2×80 | Same amplitudes/schedules | Input nonzero for eight updates; output for 79. Different random numbers remain significant protocol differences. |
| Noise streams | Input seed 901; output uses global host training RNG when `output_noise_rng` omitted | Named, isolated `forge-rng-v1` streams at protocol seed 0 | Deliberate new RNG contract; identical integer protocol seed does not imply identical streams. |
| Scoring | Clean stored particles and clean base-critic slope; live weights; 24 checks and ≥5 terminal passing checks | Same | No threshold relaxation or best-checkpoint substitution. Output-noise evaluation call count is zero for this host. |
| Runtime | CUDA, `cpu_random=false`, PyTorch `2.13.0+cu126` | CPU, one thread, runtime manifest PyTorch `2.14.0`, actual RNG receipt `2.14.0+cu130`; Python 3.14.7 | Different compatibility cohort, independently sufficient to prevent exact receipt reuse. |

At completed update 79, the historical scheduled D/particle LRs were respectively `.00005263013126336595` and `.0004444416660610054`. Current code gives the same D LR and particle LR `.0002222208330305027`, before its response gain. The historical final gain was `1.9990936517715454`; current gain is gradient-dependent, so multiplying the entire historical update trajectory by a fixed factor is not justified.

**What is a defect, and what is a changed contract?**

The historical [comparison wrapper](../../benchmarks/transfer_suite/compare_defaults.py) explicitly calls direct particles prior-owned, applies `prior_lr_mult`, and passes `particle_l2=0` to the [host wrapper](../../benchmarks/locked_shared/baseline.py). Both files are byte-identical to the pinned baseline revision. Current [BehaviorComponents.bind](../../experiments/forge/behavior_adapters.py) tags direct particles `forge_role="prior"` for scheduling but supplies only parameters to `Recipe.make_generator_optimizer`. Its table-prior branch does supply `lr * prior_lr_mult`. Construction-only verification confirms `.00425` for the direct branch and an empty optimizer state: no update was performed.

The [public Recipe API](../../particlegan/recipes.py) explicitly documents that `make_generator_optimizer` defaults to `recipe.lr`; `direct_particles` selects the response mechanism, not an automatic LR multiplier. Its higher-level `make_optimizers` adds the multiplier when assembling a prior object. Thus this is **not a proven public optimizer regression**. It is an unresolved Forge adapter policy distinction: if direct coordinates are intended to retain historical prior-role LR semantics, the binder needs an explicit multiplier; if generator-rate semantics are intended, record that protocol change and test it. Neither interpretation permits silently reusing historical PASS.

Similarly, [two_pole.train](../../benchmarks/locked_shared/two_pole.py) has always defaulted to the [locked host](../../benchmarks/legacy/locked_shared.py) L2 coefficient `.02`; the archived comparison wrapper overrode it. Forge preserves the raw host objective and explicitly blocks candidate `prior_reg` overrides through `FROZEN_HOST_RECIPE_FIELDS`. Current `.02` is source-bound and compatible with that host-owned-objective policy. The evidence establishes a historical-objective difference, **not an ignored candidate knob or proven implementation defect**. Task JSON could make the effective auxiliary coefficient easier to inspect, but changing it to chase a PASS is not warranted by this audit.

One definite provenance defect exists in the inspected snapshot: `_NamedNoise` replaces inherited generators, while inherited `NoisePolicy.receipt()` still prints `d_noise_seed=901` and `output_noise_seed=1901`. The same result's authoritative `applied.rng.bindings` records actual seeds `5961379833291764029` (critic input) and `9214932872906931917` (generator output). The output initial-state hash also matches the named stream. Those legacy seed fields are contradictory metadata. Correct them to actual named-stream identities or omit them with an explicit derivation reference; no execution failure or outcome effect is inferred from the reporting defect.

The authorized correction now overrides only `_NamedNoise.receipt()`: both seed labels come from the active generators' `initial_seed()`, `rng_derivation` records `forge-rng-v1`, the inapplicable legacy output offset is `null`, and base protocol seed remains 0. Existing v1 raw receipts remain unchanged. Their actual named manifest and execution evidence remain valid; the redundant legacy labels must not be treated as the executed stream identity. Two new construction/sampling-only tests cover enabled and disabled noise, agreement with `applied.rng`, and unchanged global/named RNG state across repeated receipt collection. Command: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_forge_behavior_adapters.py -k named_noise_receipt` — **2 passed**, 30 deselected. No model training or optimizer update was used. The coordinator will freeze and label the corrected source separately.

**Evidence limits and next action**

Keep historical and current rows in separate compatibility cohorts. Retain the negative-control FAIL and its cost without treating that ablated candidate as K3P baseline evidence. Document the direct-coordinate LR policy and add a construction-only assertion for its effective group rate; make the host L2 choice explicit in the protocol description. The metadata correction above does not resolve those historical-parity policy differences. Existing source binding already prevents these configurations from becoming the same evidence key. If a policy changes, preserve the existing receipts and bind subsequent results to the new source/task identity. No threshold, seed, or optimizer setting should be tuned from this one comparison.

Verification consisted of reading pinned Git blobs, hashing source and artifacts, comparing stored tensor bytes, constructing an unused public direct-particle optimizer, and evaluating the analytic LR scale function. The checked adapter, host, recipe, and regularizer bytes match the `a565c608…` frozen snapshot. No historical worker was executed and no optimizer `step()` or backward pass was run.

| Durable evidence | SHA-256 |
| --- | --- |
| Historical two-pole compressed bytes | `ad4b11bec9fe37566b50b2159930981e478001a2f66dc2bc4790da41ee136e53` |
| Historical two-pole decompressed artifact (summary's key) | `56ade2c803de69aea9242bd2e8717410414ec58935a58a9ef7ee689df11fe340` |
| Historical summary bytes | `bb0a6f3073b07a3074139c16111a1886478a90c8a7a7c417a7657ae90e52c743` |
| Archived [probe source](../toy100/gap-fill-20260925/sources/k3p/probe.py), matches receipt worker hash | `e8653d7e450268310d9c7fc529262279f202a2cb10b3bf1470efc7763d4452bc` |
| Historical initialization fixture hash (receipt claim; tensor bytes independently compared) | `7772e5392926ce4684bd263d3b03f02ccf866dc327746ff663e7dc01eceab230` |
| Current attempt request bytes | `5a9417f909cf5af8d99d18bfa290ea2b580261ff58ec6f6a2a19acfe5104d08c` |
| Current attempt result bytes | `dd4c3c224806a995bd7ec894e6d42249d74ad9dcf23b5735c9a4c89d205857dd` |
| Inspected v1 behavior adapter bytes, before metadata correction | `48bc1dbd2cad2248046188cd487308a87ad96ac6b1f310d51903b63fbd3aa312` |

The archived [mechanism](../toy100/gap-fill-20260925/sources/k3p/mechanism.py), [response](../toy100/gap-fill-20260925/sources/k3p/response.py), and [config](../toy100/gap-fill-20260925/sources/k3p/config.json) provide the old implementation context. The generic config's `device=cpu` is superseded by the measured result's CUDA override; actual receipt configuration and optimizer-device proof take precedence.
