"""Stdlib input/source seal before any selected RA9 numerical read."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
LANE=ROOT/'validation-cb64-ra9'
RUN=LANE/'screens/runs/grid100'
HOST=Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not (HERE/'PREPARATION-FROZEN.json').exists()
ready=ROOT/'quality/ra9/READY.json'
assert sha(ready)=='ba558fff064f5e5613656a1c1550431091efc9989f4dd34f1d85a58d17d55990'
assert sha(LANE/'source-freeze.json')=='09e6e11fd7c862d7d98fca9b5209e93be971d034ab0c6553cbd9a18bdeb6bec2'
files={}
def add(p,expected=None):
    p=Path(p);digest=sha(p)
    assert expected is None or digest==expected,p
    assert str(p) not in files or files[str(p)]==digest,p
    files[str(p)]=digest
for record,key,base in ((read(ready),'numerical_source_sha256',None),
                         (read(LANE/'source-freeze.json'),'local_sources',LANE),
                         (read(LANE/'source-freeze.json'),'external_sources',None)):
    for name,digest in record[key].items():add(Path(name) if base is None else base/name,digest)
for p in (HERE/'PROTOCOL.md',HERE/'probe.py',Path(__file__),ready,LANE/'source-freeze.json',
          LANE/'screens/source-freeze.json',ROOT/'configs/overrides-CB64-RA9.json',
          ROOT/'performance/training-regression/count-review/post-ra8-quality/grid-covariance/diagnose.py',
          ROOT/'integration/review/training-regression/post-ra4-quality/measure_saved_utils.py',
          ROOT/'integration/review/training-regression/post-ra8-quality/current-chart-resolution/probe.py',
          ROOT/'integration/review/training-regression/post-ra8-quality/current-chart-resolution/PROTOCOL.md',
          ROOT/'performance/training-regression/count-review/post-ra9-quality/local-moment-design/DESIGN.md',
          ROOT/'performance/training-regression/count-review/post-ra9-quality/local-moment-design/FROZEN.json',
          HOST/'problems.py',HOST/'toy_models.py'):
    add(p)
for name in ('final-state.pt','result.json','execution-receipt.json','job-header.json','native-fixture.json','metrics.jsonl',
             'native-clean/config.json','native-noisy/config.json','native-clean/summary.json','native-noisy/summary.json',
             'native-clean/verdict.json','native-noisy/verdict.json','native-clean/final_samples.npz','native-noisy/final_samples.npz',
             'native-clean/holdout_samples.npz','native-noisy/holdout_samples.npz'):
    add(RUN/name)
record=dict(status='FROZEN_BEFORE_NUMERICAL_LOAD',frozen_utc=datetime.now(timezone.utc).isoformat(),
    source_and_input_sha256=files,checkpoint_selection='RA9 final native grid7000 only',
    arrays='original paired20k/100k clean/noisy plus their saved target fields',
    charts=1,requested_cells=128,rank=8,chunk=256,oracle_scope='downstream annotation only',
    Torch_import=False,numerical_loads_before_seal=0,forwards_before_seal=0,
    private_fit_rng='one clone of savedCPU RNG, existing fit projection draw only',
    new_emissions=0,training_updates=0,new_seeds=0,cuda=False)
with (HERE/'PREPARATION-FROZEN.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],frozen_files=len(files),
    helper_sha256=sha(HERE/'probe.py'),freeze_sha256=sha(HERE/'PREPARATION-FROZEN.json'),frozen_utc=record['frozen_utc'])))
