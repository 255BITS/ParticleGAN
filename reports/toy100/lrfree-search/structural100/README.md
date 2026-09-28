# Native 100-Gaussian structural round (`st5` and `st6`)

This snapshot records the `st5` and `st6` experiments from the LR-free research
harness. They are experimental candidates, not changes to the public
`GANTrainer` default.
The question is whether a prior-table controller can find a settling scale
from its own motion, and whether evidence-paired particle moves can clear
bridge particles and correct mode mass without the churn seen in the earlier
birth/death arm. The fixed-noise arm isolates the effect of learned output
noise; `.029` is an attribution control, not a proposed universal setting.
`st6` tests transactional adaptive extragradient on the no-birth/death `st5`
base.

## Candidate and evaluation

| Harness candidate | Initialization / mechanism | Output noise | Overrides |
|---|---|---|---|
| `st5-scaleaware-qr` | **QR primary** / two-scale, scale-aware settle; no BD | learnable | [config](configs/st5-scaleaware-qr.json) |
| `st5-scaleaware-xv` | Xavier control / same, no BD | learnable | [config](configs/st5-scaleaware-xv.json) |
| `st5-scaleaware-pair-xv` | Xavier control / significant excess-deficit BD pairs | learnable | [config](configs/st5-scaleaware-pair-xv.json) |
| `st5-scaleaware-pair-fixed-xv` | Xavier control / paired BD | fixed `.029` | [config](configs/st5-scaleaware-pair-fixed-xv.json) |
| `st6-eg-qr` | **QR primary** / no BD, transactional adaptive extragradient | learnable | [config](configs/st6-eg-qr.json) |

The library's `batch_feature_zero` QR initialization is the **primary** native
comparison. The earlier `initialization: null` Xavier runs are secondary
controls and must not be ranked against QR rows; the earlier Xavier-first
interpretation is superseded. Every registered arm uses noisy scoring
(`eval_output_noise: true`). Standard native cards run for 7,000 updates;
the rotated extension runs for 14,000. Both use the frozen scorer. The `st5`
source [patch](st5-vs-dv12-st.patch) is against the harness's
`candidates/dv12-st/package`.
It changes only `particlegan/continuous.py` and `particlegan/birth_death.py`:
row-equal cosine evidence, two-scale tests with nominal per-window alpha
spending, scale-aware reversal, row-lineage invalidation after cloning, and
birth/death moves only between supported excess and deficit rows. The
per-window test level is not an anytime error guarantee; the kNN/BH null
is approximate for dependent adaptive particle clouds.

The [st6 patch](st6-vs-st5.patch) changes `recipes.py` and `training.py`
relative to `st5`. It adds a checkpointed predictor/corrector update with
one committed optimizer step. CPU transaction and RNG receipt tests passed;
the early native diagnostic below is not a 7,000-update verdict.

## QR native results (primary)

The frozen native coverage and accuracy verdict is **FAIL** for all three
`st5-scaleaware-qr` cards. Each card had 34 observations and zero passing
checks. The table gives step-7,000 live noisy metrics to explain the failure;
the formal verdict also requires the final five observations and independent
100k holdout. Precision must be at least `.97`, covariance eigenvalue ratios
must stay within `.40–1.70`, centre RMS at most `.20σ`, and radial KS at most
`.04`.

| Native task | Frozen verdict | Modes | Precision | Covariance eigenvalue range | Centre RMS | Trace bias | Radial KS | Learned σ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| grid100 | FAIL | 100 | .979 | .163–2.432 | .364σ | +.074 | .084 | .00677 |
| rotated100 | FAIL | 100 | .962 | .059–1.857 | .281σ | −.261 | .120 | .00152 |
| staggered100 | FAIL | 100 | .960 | .034–2.123 | .522σ | −.249 | .067 | .00329 |

Grid reaches all 100 modes and clears the precision limit, but its modes are
streaked and off-centre. Rotated and staggered also reach all modes but miss
precision and shape. Learned output noise shrinks far below the data σ of `.03`. Thus
the scale-aware settle rule alone does not solve native shape. The 13 harness
gates have **not** been run on this QR candidate; this table makes no claim
about them.

The exact final, holdout, and gate fields are archived in
[native100-results.json](native100-results.json). The research harness's
`native100-diagnostics.jsonl` is a read-only sidecar: it measures bridge
outliers, per-mode geometry, row motion, and settle decisions from the same
evaluation clouds without changing the frozen scorer or verdict.

### Rotated100 extension to 14,000 updates

`st5-scaleaware-qr-14k` keeps the same package, QR initialization, recipe,
and seed; the candidate option `native_steps=14000` extends only the evaluator
budget. Its first 7,000 update rates and diagnostics are byte-identical to
`st5-scaleaware-qr`. The first 34 metric rows match after excluding elapsed
seconds, and the 7k noisy subbudget holdout matches the original. The frozen
7k subbudget remains **FAIL**.

At 14k the frozen native result is still **FAIL**, with 0/62 passing checks,
no terminal streak, and a failing independent holdout. All observations below
cover 100 modes; values are live noisy measurements.

| Step | Precision | Covariance eigenvalue range | Centre RMS | Trace bias | Radial KS | Learned σ |
|---:|---:|---:|---:|---:|---:|---:|
| 7,000 | .962 | .059–1.857 | .281σ | −.261 | .120 | .00152 |
| 10,000 | .974 | .086–1.738 | .203σ | −.183 | .081 | .00142 |
| 11,000 | .973 | .109–1.696 | .223σ | −.143 | .057 | .00140 |
| 14,000 | .975 | .066–2.741 | .308σ | −.189 | .069 | .00135 |

The prior LR cut from `.00425` to `.002125` at step 8,088, then DRIFT
restored `.00425` at 11,160 and `.0085` at 11,928. Precision and radial KS
improved, yet no observation passed: even the best observed minimum
eigenvalue ratio (`.143`), minimum radial median (`.560`), absolute trace
bias (`.143`), and radial KS (`.0548`) remained beyond their limits. The
14k holdout has precision `.97193` but centre error `.298σ`, trace bias
`−.18264`, and radial KS `.06925`. Extra time alone did not settle the
shape. The exact 14k gate, 7k subbudget, holdout, and selected trajectory are
in [native100-results.json](native100-results.json).

`st6-eg-qr` was stopped after the grid diagnostic deteriorated: at step 500
it had zero covered modes and HQ fraction `.0032`. Its harness `result.json`
says `ERROR` because no final native result was produced. That is an early
rejection signal, **not** a frozen native FAIL or PASS verdict. No st6 QR
rotated, staggered, or 13-gate verdict is claimed.

## Evidence from earlier birth/death arms

These **Xavier-control** results are secondary to QR. The native sidecar and
repeated-index snapshots in the research harness's
`reports/st3-bd-native-evidence.md` show that unrestricted `st2` moves damaged
coverage on all three 100-Gaussian cards. Removing neutral-parent moves in
`st3` greatly reduced churn, but did not protect mode coverage.
The `st3` no-BD control reproduces the `st2` no-BD grid and rotated
trajectories through step 7,000.
All rows below have a frozen **FAIL** result at 7,000 updates; their final
checkpoint metrics describe the failure rather than replace the formal gate.

| Arm / task | Step | Modes | Precision | Mass TV | BD moves (neutral parents) |
|---|---:|---:|---:|---:|---:|
| `st2` no BD / rotated | 7,000 | 100 | .930 | .030 | 0 |
| `st2` unrestricted BD / rotated | 7,000 | 93 | .827 | .118 | 396,158 (335,814) |
| `st2` no BD / grid | 7,000 | 98 | .969 | .062 | 0 |
| `st2` unrestricted BD / grid | 7,000 | 89 | .910 | .157 | 513,014 (460,328) |
| `st2` no BD / staggered | 7,000 | 100 | .950 | .040 | 0 |
| `st2` unrestricted BD / staggered | 7,000 | 91 | .887 | .132 | 425,397 (372,725) |
| `st3` paired BD / grid | 7,000 | 94 | .927 | .103 | 74,038 (0) |
| `st3` paired BD / rotated | 7,000 | 97 | .849 | .074 | — |
| `st3` paired BD / staggered | 7,000 | 91 | .880 | .131 | — |
| `st4` paired BD / grid | 7,000 | 95 | .949 | .099 | — |
| `st4` paired BD / rotated | 7,000 | 97 | .849 | .074 | 56,328 (0) |
| `st4` paired BD / staggered | 7,000 | 91 | .880 | .131 | — |
| `st5` paired BD / grid | 7,000 | 95 | .949 | .099 | — |
| `st5` paired BD / rotated | 7,000 | 99 | .838 | .078 | — |
| `st5` paired BD / staggered | 7,000 | 91 | .880 | .131 | — |

`—` means a final move count was not tabulated in this report. The
mode, precision, and mass values come from each frozen `result.json`; the
move counts shown are from the research evidence note.

In the `st3` paired rotated run, mode 43 lost noisy HQ mass (122 to 92 of
20,000 samples) over steps 1,750–2,000. In the repeated-index clean
snapshots, 13 HQ rows left and five arrived; several leavers landed in modes
that already had more mass than mode 43. On staggered, paired moves also
pushed some rows outside their original mode's 3σ HQ ball. The likely gap is
that BD certifies *local density* at a particle centre while the gate needs
enough total HQ mass in each mode. Snapshot indices locate transfers over an
interval; they do not identify the exact `_move` events, so the causal
assignment to BD remains limited by intervening gradient updates.

## Verdict scope

| Candidate | Native grid / rotated / staggered | 13 harness gates |
|---|---|---|
| `st5-scaleaware-qr` | FAIL / FAIL / FAIL, each 0/34; holdouts FAIL | Not run |
| `st5-scaleaware-qr-14k` | Not run / rotated FAIL 0/62, holdout FAIL / not run | Not run |
| `st5-scaleaware-pair-xv` (Xavier control) | FAIL / FAIL / FAIL | Not a QR comparison |
| `st6-eg-qr` | Grid stopped at step 500; no formal verdict / not run / not run | Not run |

No `st5` candidate has a complete passing native three-task set. Formal
promotion would also require the 13 noisy-scored harness gates.

The source identity for this snapshot is:

| Item | SHA-256 |
|---|---|
| Base `continuous.py` | `481d1398015028a91885f5b13d8654cbd8216e47098c5dbc01e388f8514735ed` |
| Base `birth_death.py` | `8d71f159f2c2106aeeaeea0c192ba269818c85bfea17e0df0530f6879eb3ef11` |
| Patched `continuous.py` | `f7aa28aca51f7e57b15ac4d8b6e1f0ba764f25ac65a9bc4713b94e0723fdfadd` |
| Patched `birth_death.py` | `72bbe5bc7d2445e87d265dd21428128cbc0c80c77f82bf17d27c14433b4e2f13` |
| `st5` harness package digest (all `particlegan/*.py`) | `09d67af076169b1118b1a565faba9846fec865d5e7d81944a9b0a5f4e22884b5` |
| `st6` patched `recipes.py` | `c863c0bc261d8f1d0a6ef16772622a6340b08de2c729da1eee2d29760f1b3cdb` |
| `st6` patched `training.py` | `f0c55a0d89cc01037109f864b7bf699f0c15befc500902b2e87e9ef1b14b818b` |
| `st6` harness package digest | `96e2c7e10cc1ec913ff6d2f918a212c8f17f158fe487a34163d0407515341b31` |

## Reproduce in the research harness

The commands below make isolated `st5` and `st6` packages from the recorded
base, then submit only the primary QR `st5` native run. They require the
research harness at `/ml2/hypergan/lrfree-20260926` and its Python
environment. Use a new candidate name so the recorded verdict remains intact.

```bash
HARNESS=/ml2/hypergan/lrfree-20260926
ARTIFACT=/ml2/hypergan/ParticleGAN-k3p-continuous-search/reports/toy100/lrfree-search/structural100
REPRO_DIR=$(mktemp -d)
cp -a "$HARNESS/candidates/dv12-st/package" "$REPRO_DIR/package"
patch -p1 -d "$REPRO_DIR/package" < "$ARTIFACT/st5-vs-dv12-st.patch"
cp -a "$REPRO_DIR/package" "$REPRO_DIR/st6-package"
patch -p1 -d "$REPRO_DIR/st6-package" < "$ARTIFACT/st6-vs-st5.patch"

cd "$HARNESS"
/tmp/pr38-default-env/bin/python harness/submit.py \
  --cand st5-scaleaware-qr-repro --package-root "$REPRO_DIR/package" \
  --overrides "$ARTIFACT/configs/st5-scaleaware-qr.json" \
  --candidate-options '{"eval_output_noise":true}' --tasks native \
  --note 'st5 QR structural100 reproduction'
tail -f pool.log
```

The Xavier and fixed-noise configs in this folder are optional attribution
controls. The early-stopped EG arm used `$REPRO_DIR/st6-package`,
`configs/st6-eg-qr.json`, and the same noisy candidate option; it has no
formal native verdict to reproduce.

To reproduce the 14k rotated extension with the same QR overrides and
candidate options, submit this from the harness directory after constructing
`$REPRO_DIR/package` above:

```bash
/tmp/pr38-default-env/bin/python harness/submit.py \
  --cand st5-scaleaware-qr-14k-repro --package-root "$REPRO_DIR/package" \
  --overrides "$ARTIFACT/configs/st5-scaleaware-qr.json" \
  --candidate-options '{"eval_output_noise":true,"native_steps":14000,"save_final_state":true}' \
  --tasks rotated100 --note 'st5 QR rotated100 14k extension'
```

Native observations are in `runs/<candidate>/<task>/metrics.jsonl`; the
separate `native100-diagnostics.jsonl` describes bridge outliers, per-mode
shape and centre error, prior motion, settle decisions, and particle moves.
`rates.jsonl` records applied learning rates and output sigma. Read the
frozen `result.json` verdicts, including the final five observations and the
independent 100k holdout, before claiming a pass. The 13 harness gates
(10 quick tasks, `img_intensity2` at 1,200 updates, ring and stationary)
must also be checked before promoting this candidate into core code.

The working research notes are `lrfree-20260926/reports/st3-100g-round.md`,
`lrfree-20260926/reports/st3-bd-native-evidence.md`, and
`lrfree-20260926/reports/native100-metrics.md`. This PR snapshot records
the complete QR native failure and the early EG stop; the 13-gate suite is
still untested for these candidates. No pass or core-promotion claim follows.
