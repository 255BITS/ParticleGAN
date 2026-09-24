Run from the assigned repository. Outputs below must not already exist. Read
the runtime supervisor before each batch and obey STOP. These commands reproduce
diagnostics; none authorize full1200/native100 promotion.

```bash
cd /ml2/hypergan/gan-attempts/g075-20260924T180736Z/own_state_stability/20260924T180736Z-1586957/repo
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES=''
P=/tmp/pr38-default-env/bin/python
R=reports/toy100/h_stability
A=$R/adam-response
GATES='mode_hold two_pole unipolar mid_scale_identity cover_leftover trajectory residual_student img_stripes2 img_bars4 vector_overlap img_blobs4 img_intensity2 vector_unequal_mass vector_unequal_width'
```

Best candidate: warm diagnostic, fresh cold acquisition, then its own200.

```bash
cat ../supervisor.md
$P -u "$R/adam_response_probe.py" --variant eps_net_1m --output "$A/replay-best/warm" --ledger "$A/replay-best-tests.jsonl"
cat ../supervisor.md
$P -u "$R/adam_response_cold.py" --declaration "$A/best-diagnostic-declaration.json" --output "$A/replay-best/cold" --ledger "$A/replay-best-tests.jsonl" --workers 1 --tasks $GATES
cat ../supervisor.md
$P -u "$R/adam_response_probe.py" --variant eps_net_1m --cold-candidate "$A/replay-best/cold/eps_net_1m" --output "$A/replay-best/own200" --ledger "$A/replay-best-tests.jsonl"
```

All nine initial warm diagnostics, grouped exactly as proposed (sequential
workers reproduce the fixed RNG streams). Every probe stops on its first failure.

```bash
cat ../supervisor.md
for C in eps_all_1m eps_gp_1m eps_d_1m; do
  $P -u "$R/adam_response_probe.py" --variant "$C" --output "$A/replay-batch1/$C" --ledger "$A/replay-all-tests.jsonl"
done
cat ../supervisor.md
for C in tensor_g tensor_gp tensor_all; do
  $P -u "$R/adam_response_probe.py" --variant "$C" --output "$A/replay-batch2/$C" --ledger "$A/replay-all-tests.jsonl"
done
cat ../supervisor.md
for C in eps_net_1m split_prior eps_net_split_prior; do
  $P -u "$R/adam_response_probe.py" --variant "$C" --output "$A/replay-batch3/$C" --ledger "$A/replay-all-tests.jsonl"
done
```

Cold gates and the remaining explicit cheap diagnostics:

```bash
cat ../supervisor.md
$P -u "$R/adam_response_cold.py" --declaration "$A/batch1/cold-declaration.json" --output "$A/replay-cold1" --ledger "$A/replay-all-tests.jsonl" --workers 2 --tasks $GATES
cat ../supervisor.md
$P -u "$R/adam_response_probe.py" --variant eps_all_1m --cold-candidate "$A/replay-cold1/eps_all_1m" --output "$A/replay-eps-all-own200" --ledger "$A/replay-all-tests.jsonl"
cat ../supervisor.md
$P -u "$R/adam_response_cold.py" --declaration "$A/batch3/declaration.json" --output "$A/replay-cold3" --ledger "$A/replay-all-tests.jsonl" --workers 3 --tasks $GATES
cat ../supervisor.md
$P -u "$R/adam_response_probe.py" --variant eps_net_1m --cold-candidate "$A/replay-cold3/eps_net_1m" --output "$A/replay-eps-net-own200" --ledger "$A/replay-all-tests.jsonl"
cat ../supervisor.md
$P -u "$R/adam_response_cold.py" --declaration "$A/batch3/mobility-diagnostic-declaration.json" --output "$A/replay-mobility-diagnostic" --ledger "$A/replay-mobility-raw.jsonl" --workers 2 --tasks two_pole
```

The last command is only a diagnostic after ring failure. When importing its
raw ledger, name the gate `diagnostic_two_pole80_after_ring_fail`, as this attempt
did. Do not combine it with another policy's acquisition.

Final-code relevant tests, each distinct case once (34 tests). Historical run
totals 22+16+20=58 include necessary repeats after code changes.

```bash
$P -m pytest -q tests/test_adam_response.py tests/test_continuous_candidates.py tests/test_continuous_probe.py tests/test_locked_shared_behavior.py --junitxml="$A/replay-regression.xml"
```

Earlier exact test selections and source versions: `regression1.log/xml` used
continuous_candidates, continuous_probe, locked_shared_behavior; `regression2`
and `regression3` used adam_response and continuous_candidates. The corresponding
batch source snapshots retain the proposal code executed at that stage.
