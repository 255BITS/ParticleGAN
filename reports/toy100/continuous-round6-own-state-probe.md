# Own-acquired continuation prepared for sample-anchor candidates

The [source-bound driver](sample_anchor_own_state_probe.py) is ready but has not run a training continuation. Its explicit `--factory` must match the qualified cold declaration exactly, whether the rest-guard or pre-G-start candidate survives. It rejects a missing, failed, mismatched, or modified cold trajectory/ring before creating an output directory. The factory generated host, saved full model/Adam/EMA/RNG/noise state, constant-rate configuration, and original 1200-update noise horizon are checked against the cold artifact.

After a qualified cold ring, `--phase hold` restores its own update-1200 snapshot before `set_step`, then reuses the reviewed exact resumer for updates 1201–2400. Every completed update is evaluated with the original eight-mode/HQ 0.9 thresholds; the first failure stops the continuation. The driver records actual Adam calls, constant rates, full RNG/noise history, clean functional movement, generated resumed source, and a complete post-checkpoint state. An early failure remains a failed diagnostic, not a complete hold.

Only a 1200/1200 passing hold can enable `--phase response`. The driver restores that own-acquired update-2400 state, runs paired unperturbed and fixed `+0.35` G output-bias branches over absolute updates 2401–2450, and evaluates a frozen perturbed control on the same unchanged target and absolute noise clock. It uses the existing five-check suffix response criterion and saves both trained final states. There is no LR decay, moment reset, noise-horizon restart, target shift, or oracle use in selection. A 50-update response miss is a local filter result, not a claim that recovery is impossible.

No-training preflight tests cover both factory bindings, the exact AST resume edit, restoration of the factory patch, and rejection of incomplete cold or hold artifacts before training. They pass 3/3 in 3.18 seconds; the [test log and source hashes](continuous-evidence/round6-own-state-preflight/manifest.json) preserve that check.

When the cold gate is complete, invoke the hold with the frozen cold artifact, then supply the passing hold directory for response:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -m reports.toy100.sample_anchor_own_state_probe --phase hold --factory reports.toy100.sample_anchor_rest_candidate:sample_anchor_rest_candidate --cold PATH_TO_QUALIFIED_COLD --output PATH_TO_OWN_HOLD
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -m reports.toy100.sample_anchor_own_state_probe --phase response --factory reports.toy100.sample_anchor_rest_candidate:sample_anchor_rest_candidate --cold PATH_TO_QUALIFIED_COLD --hold PATH_TO_OWN_HOLD --output PATH_TO_RESPONSE
```

If the pre-G-start candidate passes cold instead, use `reports.toy100.sample_anchor_prestart_candidate:sample_anchor_prestart_candidate` in both commands. A factory/cold declaration mismatch fails before either run.
