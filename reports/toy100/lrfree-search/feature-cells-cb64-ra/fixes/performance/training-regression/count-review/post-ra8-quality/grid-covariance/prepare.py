import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
RUN = ROOT / 'validation-cb64-ra8/screens/runs/grid100'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')
HOST = Path('/ml2/hypergan/ParticleGAN-k3p-continuous-search')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (HERE / 'PREPARATION-FROZEN.json').exists()
paths = [HERE/'PROTOCOL.md', HERE/'diagnose.py', Path(__file__),
         ROOT/'quality/ra8/READY.json', ROOT/'quality/ra8/COMPOSITION.json',
         ROOT/'validation-cb64-ra8/source-freeze.json', ROOT/'configs/overrides-CB64-RA8.json',
         HARNESS/'screen.py', HARNESS/'native100_score.py', HARNESS/'tasks/native100_fixture.json',
         HARNESS/'hosts/native100/problems.py', HARNESS/'hosts/native100/metrics.py',
         HARNESS/'hosts/native100/accuracy.py', HOST/'lib/toy_models.py']
paths += sorted((ROOT/'pkg-CB64-RA8/particlegan').rglob('*.py'))
paths += [RUN/p for p in ('final-state.pt','result.json','execution-receipt.json',
                         'metrics.jsonl','native100-diagnostics.jsonl','rates.jsonl',
                         'native-fixture.json','job-header.json')]
for kind in ('clean','noisy'):
    paths += [RUN/f'native-{kind}'/p for p in ('final_samples.npz','holdout_samples.npz',
                                             'events.jsonl','summary.json','verdict.json','config.json')]
out = dict(status='FROZEN_BEFORE_MEASUREMENT', source_and_input_sha256={str(p):sha(p) for p in paths},
           scope='Fixed saved-only covariance/lineage/preprojection chart diagnostic; original gates unchanged')
(HERE/'PREPARATION-FROZEN.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'], files=len(paths), sha256=sha(HERE/'PREPARATION-FROZEN.json'))))
