"""Seal a source-only reserved design; never load/hash numerical PT inputs."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not (HERE/'FROZEN.json').exists()
ready=read(ROOT/'quality/ra9/READY.json')
assert sha(ROOT/'quality/ra9/READY.json')=='ba558fff064f5e5613656a1c1550431091efc9989f4dd34f1d85a58d17d55990'
sources={}
for name in ('feature_cells.py','birth_death.py','training.py','anchor_birth.py','birth_phase.py'):
    p=ROOT/'pkg-CB64-RA9/particlegan'/name
    assert sha(p)==ready['package_source_sha256'][name]
    ast.parse(p.read_text())
    sources[str(p)]=sha(p)
for name in ('quality/ra9/READY.json','validation-cb64-ra9/source-freeze.json',
             'performance/training-regression/count-review/post-ra8-quality/grid-covariance/REPORT.md',
             'performance/training-regression/count-review/post-ra8-quality/grid-covariance/FROZEN.json',
             'performance/training-regression/count-review/post-ra8-quality/grid-neighbor-subset/REPORT.md',
             'performance/training-regression/count-review/post-ra8-quality/grid-neighbor-subset/FROZEN.json',
             'performance/sampler-regression/cpu-plan-review/post-ra8-quality/grid-saved-diagnostic/REPORT.md',
             'performance/sampler-regression/cpu-plan-review/post-ra8-quality/grid-saved-diagnostic/FROZEN.json',
             'integration/review/training-regression/post-ra8-quality/current-chart-resolution/probe.py',
             'integration/review/training-regression/post-ra8-quality/current-chart-resolution/PROTOCOL.md'):
    p=ROOT/name;sources[str(p)]=sha(p)
receipt=dict(status='VALID_SOURCE_ONLY_RESERVED_DESIGN',scope='Critique and one deferred split-reference local-moment feasibility diagnostic',
    candidate_change_implemented=False,diagnostic_executed=False,production_law_qualified=False,
    trigger='Parent selection only after final original RA9 grid verdict',
    numerical_PT_loads=0,numerical_PT_hashes=0,numerical_measurements=0,Torch_import=False,
    model_forwards=0,random_draws=0,new_seeds=0,training_updates=0,cuda=False,
    oracle_production_inputs=False,quality_verdict=None,source_sha256=sources,
    major_limits=['Trained-D/shared-FIFO dependence','Multimode aliasing and conditional group selection',
        'Global moment rejection does not certify all local movements','Latent clean and emitted laws differ',
        'Moment transport needs its own law; existing count certificates do not authorize it'])
with (HERE/'receipt.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
files={str(p):sha(p) for p in (HERE/'DESIGN.md',HERE/'receipt.json',Path(__file__))}
frozen=dict(status=receipt['status'],frozen_utc=datetime.now(timezone.utc).isoformat(),
    files=files,source_sha256=sources,quality_verdict=None,executed=False)
with (HERE/'FROZEN.json').open('x') as f:f.write(json.dumps(frozen,indent=2)+'\n')
for name,digest in {**files,**sources}.items():assert sha(name)==digest,name
print(json.dumps(dict(status=receipt['status'],receipt_sha256=sha(HERE/'receipt.json'),freeze_sha256=sha(HERE/'FROZEN.json'))))
