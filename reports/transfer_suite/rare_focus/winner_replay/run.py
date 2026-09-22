import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.transfer_suite import suite, vector_tasks
from benchmarks.transfer_suite.linear_skip_refinement_research import constructor
from benchmarks.transfer_suite.protocol import test_verdict

OUT = Path(__file__).resolve().parent
REFERENCE = Path('/tmp/pr36-valid-linear-final/screen/episodes/linear_skip_d96_beta5__vector_unequal_mass.json.gz')


def write(name, value):
    (OUT/name).write_bytes(gzip.compress((json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode(), mtime=0))


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if k not in ('seconds', 'controller_seconds', 'confirmed_seconds', 'stable_from_seconds')}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


protocol = suite.snapshot(OUT)
protocol['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
write('protocol.json.gz', protocol)
reference = json.loads(gzip.decompress(REFERENCE.read_bytes()))
(OUT/'reference.json.gz').write_bytes(REFERENCE.read_bytes())
with patch.object(vector_tasks, 'SimpleMLPDiscriminator', constructor(reference['candidate']['architecture'])):
    result = vector_tasks.run_episode(reference['spec'], reference['policy'], fixed=True)
write('result.json.gz', result)
assert clean(result) == clean(reference['result'])
verdict = test_verdict(reference['spec'], result)
assert verdict['passed'] and verdict['convergence']['passing_suffix'] == 6
suite.verify_source(protocol)
write('parity.json.gz', dict(entire_result_except_timing_exact=True, all_24_live_ema_and_actions_exact=True,
                             reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(), verdict=verdict))
print('PASS exact replay; suffix', verdict['convergence']['passing_suffix'], 'confirmed', verdict['convergence']['confirmed_step'])
