#!/bin/bash
# CPU-only regression run for pkg-E19 (the final package): the support-test tests, the reviewer's extra tests, the feature-scale (gauge) tests, the high-dimension parent test, and the
# earlier E12-E14 suites against the same package where they take a package argument. Every script prints [ok]/[FAIL] lines and ALLOK/FAILED; grep for FAIL.
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 NT=2
P=/tmp/pr38-default-env/bin/python; T=/ml2/hypergan/gan-attempts/noout-20260928/tests; E19=/ml2/hypergan/gan-attempts/noout-20260928/pkg-E19
for f in "$T/test_E15.py $E19" "$T/test_E17_extra.py $E19" "$T/test_E18.py $E19" "$T/test_E19.py $E19" "$T/test_E14.py" "$T/test_E13.py" "$T/test_E12.py" "$T/review2/serve_symmetry_test.py $E19" "$T/review2/bh_check.py" "$T/review2/leak_probe.py $E19 450" "$T/review2/failed_load_probe.py $E19"; do
  echo "=== $f"; timeout 1800 $P $f 2>&1 | grep -v Warning | grep -v "p = (1" | tail -n 30
done
