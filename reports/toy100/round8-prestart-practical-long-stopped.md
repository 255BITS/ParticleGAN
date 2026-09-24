# Prestart anchor long continuation stopped on scope change

The [separate practical observer](sample_anchor_practical_long_probe.py) resumed the **own-acquired** post-update-2400 state with the frozen prestart anchor method. It retained the same target, G/D/prior nominal rates `.00425/.00425/.0085`, Adam/EMA/RNG state, and original 1200-step noise horizon. Unlike the earlier strict first-failure observer, this one would continue through finite quality dips and count every failure/recovery run. Its source was frozen before launch; two no-training grade tests passed, and a 50-update preflight reproduced all archived unperturbed response checkpoints exactly.

The user then narrowed the objective to fixes of the **GAN formulation itself**. This anchor method adds a separate coverage target, so the run was intentionally interrupted. The final classification is **`STOPPED_SCOPE_CHANGE`**. The last complete update is **9488**, with **7088/7088** resumed every-update checks at eight modes/HQ 1 and no observed failure run. The interrupt arrived during the SVD inside update 9489. The driver’s generic exception artifact says `ERROR_INCOMPLETE`/`KeyboardInterrupt`; the [separate stop receipt](continuous-evidence/round8-prestart-practical-long-stopped/stop-status.json) records the intentional cause. Its saved pre-`set_step(9489)` state is a complete update-9488 boundary: both Adam step counters and the noise clock equal 9488. A distinct mid-update partial snapshot is marked nonrestartable. **This is neither a numerical instability nor a 12,000-update pass.** Runtime to interruption was 509.56 s on one CPU thread.

Passive state summaries show why quality alone would be too weak a stability claim. The following are Euclidean norms of stored model parameters and Adam first moments; endpoints come from exact state snapshots, intermediate rows from the read-only 200-update observer.

| Complete update | G params | D params | prior params | D Adam first moment | G/prior Adam first moment |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2400 | 11.0957 | 12.6624 | 3.8575 | .25585 | .27032 |
| 2600 | 11.0920 | 12.7806 | 3.8625 | .27839 | .53257 |
| 6000 | 11.0435 | 14.7247 | 3.9812 | .11486 | .16283 |
| 9400 | 11.0203 | 15.6402 | 4.1412 | .16618 | .26280 |
| 9488 | 11.0195 | 16.1987 | 4.1445 | .33023 | .22645 |

All inspected model/Adam tensors are finite. The D parameter norm increased by about 28% from 2400 to 9488 while clean coverage remained perfect; this is measured critic motion, neither a proof of divergence nor bounded-parameter convergence. D Adam second-moment norm changed `.04189→.06116`; G/prior `.01394→.01258`. All 35 periodic GN summaries (2600–9400) report thin-J rank 24; median raw singular-value ratio rises from 16.29 to 18.30. These finite chart measurements do not bound future dynamics or every internal game direction.

The [archive](continuous-evidence/round8-prestart-practical-long-stopped/manifest.json) includes all 7088 compact quality rows, 35 passive state summaries, exact input/safe/partial snapshots, the interrupt transcript, generated host, full source bytes and SHA-256 manifest. The first 50 compact grades match the archived unperturbed response control. The original strict omission test remains failed at 2401–2402 in its separate branch; its measured ordinary-data recovery by 2403 remains a separate result. This method also has known mass and within-mode spread mismatch, and it is outside the user’s present GAN-only candidate scope.
