# Forge technique roster

The current executable catalog contains 12 idea declarations. The inventory
uses the ordinary `discriminator_stability` view: **3 smoke, 19 quality and 2
endurance requirements**. Each tier's denominator stays fixed when execution
stops or a host is unsupported. Architecture-transfer and paired-sampling
diagnostics are reported separately. The screening profile remains provisional.

| Technique | Idea ID | Actual formulation | Ordinary applicability |
| --- | --- | --- | --- |
| R1/R2 baseline, matched | `k3p-r1r2-matched-v1` | Fixed squared-L2 real/fake penalty, coefficient 1; K3P optimizer | Execute eligible tiers |
| BCap, matched | `k3p-bcap-matched-v1` | Fixed one-sided L2 real/fake cap, coefficient 1, cap 1; K3P optimizer | Execute eligible tiers |
| K3P | `k3p` | Explicit K3P penalty and optimizer | Execute eligible tiers |
| KA2 | `ka2` | Explicit KA2 penalty and optimizer; current public default | Execute eligible tiers |
| K3P without critic anchor | `forge-onboarding-anchor-ablation` | K3P with `reg_anchor_weight=0` | Execute eligible tiers |
| K3P without critic penalty | `forge-no-critic-penalty` | K3P with `reg_coeff=0` | Execute eligible tiers |
| K3P without A2 | `k3p-a2-off-native-diagnostic` | K3P with `latent_damping_max_rate=0` | Execute eligible tiers |
| K3P without training output noise | `k3p-no-output-noise-diagnostic` | K3P with `output_noise_std=0` | Execute eligible tiers |
| Released v0.7 GAN v3, matched MoG | `release07-gan-v3-mog-v1` | Full released BCap coefficient 6/cap 1.25 recipe; matched host adaptation | BLOCKED at smoke by frozen host ownership |
| Released v0.7 GAN v3, original cloud host | `release07-gan-v3-cloud-v1` | Full released BCap coefficient 6/cap 1.25 recipe; original cloud diagnostic host | BLOCKED at smoke by frozen host ownership |
| E22 | `e22` | KA2 plus DV12/stationarity, row controls, learned noise and state-selected serving | BLOCKED: no frozen policy-aware tasks |
| Atlas | `atlas` | E22 plus automatic feature cells and settled reopen guard | BLOCKED: no frozen policy-aware tasks |

The R1/R2 row is the standard zero-centered penalty
`coefficient/2 * (E ||grad D(real)||² + E ||grad D(fake)||²)`.
It is a matched penalty baseline retaining the K3P optimizer, A2, noise,
schedule and task host. It does not represent a separately established plain
Adam recipe or import earlier coefficient .02/.1 results. BCap uses the same
matched settings and replaces squared gradient norm by squared excess above
its cap. These fixed arms ignore the K3P penalty's blend and EMA anchor.

The released v0.7 rows preserve the full effective recipe rather than aliasing
the coefficient-1 matched BCap row. Their complete declarations pin dimensions,
batch size and other task-owned fields. Read-only preflight supports the three
ordinary native tasks individually but blocks the other 21 tasks, including all
smoke requirements. Their separately frozen native diagnostic hosts remain
available in `formulation_comparison`; ordinary gates confer no permission to
skip blocked smoke. The prior is task-owned: a card's cloud label does not
force cloud sampling on another task.

## Recipe binding repair

An idea's `parent` records provenance; it does not inherit runtime recipe
fields. After the public default became KA2, the four K3P ablation declarations
silently resolved to KA2 despite their K3P names and hypotheses. This inventory
pins `name=k3p` and `critic_formulation=k3p` in those declarations. Each now
differs from the explicit K3P reference in exactly its stated mechanism field.
This creates truthful new declaration identities without changing historical
receipts, scientific verdicts, sampling laws or thresholds.

The matched R1/R2 and BCap declarations already select K3P through `reg_arm`;
the resolved recipe's label alone does not determine its optimizer. Focused
contract checks in [test_forge_technique_bindings.py](../../tests/test_forge_technique_bindings.py)
verify the actual optimizer/penalty factories, one-factor ablations and policy
blockers without training.

## Other existing research

Historical techniques remain searchable in the
[compiled experiment memory](EXPERIMENT_MEMORY.md) and existing Forge boards.
They require an explicit public formulation/host declaration before becoming a
current executable row. The current roster excludes these families from new
training:

- Legacy `c_eikonal`, `d_asym`, `e_interp`, `g_interp_cap`, alternative norms and
  finite-difference BCap: implementations live in `benchmarks/legacy`; the
  current public penalty supports K3P, fixed R1/R2 and fixed BCap only.
- Legacy loss choices, K3/K3G, RG5, direct-particle response, handoff/prior
  coupling, stationarity/continuous research arms and initialization searches:
  recorded under their original loops, recipes, hosts and serving laws.
  Existing reports do not define interchangeable current Forge candidates.
- E4 and E14–E22 interventions, LR/horizon arms, D-tracking floors, prior EMA,
  feature-gauge and small-batch studies: retained in the
  [31-arm historical archive](supplemental/pr155-current-archive-v1/README.md).
  This archive records 93 native receipts, including 11 noisy failures and
  clean failures for all 93; horizons and sources are distinct.
- Public AE/VAE/DDGAN/MoG names select model/prior components rather than new
  critic techniques. Forge hosts already declare their component choices.
  `e22_routed` is a separate caller-owned conditional policy adaptation and
  has no current Forge task/idea declaration.

E22's archived three native confirmations, E19a's 13-host portability evidence
and the [original Atlas 19/19 CUDA replay](../develop-gates-20261001/README.md)
remain separate historical cloud/noisy cohorts. The three Atlas clean native
diagnostics still fail. Neither original positives nor the legacy BCap 22-host
results fill current learned-MoG/clean-live tier cells. CPU/CUDA and runtime
cohorts remain distinct as well.

Expand the inventory by registering a truthful public-API idea, declaring any
needed task contract, and running Forge within an explicit budget. Regenerate
the board from receipts; retain missing and blocked cells, avoid seed-only
variants, and require every ordinary lower tier before spending on the next.
