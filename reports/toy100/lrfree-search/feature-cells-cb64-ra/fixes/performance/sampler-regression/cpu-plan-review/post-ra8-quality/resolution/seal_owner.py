"""Post-exit owner seal; no numerical inputs or Torch are loaded."""
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
AREA=Path(__file__).resolve().parent
PACKAGE=AREA/'pkg-RESOLUTION'
BASE=ROOT/'pkg-CB64-RA8'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name,value):
    with (AREA/name).open('x') as handle:handle.write(json.dumps(value,indent=2)+'\n')


for name in ('SOURCE-FROZEN.json','TESTS-FROZEN.json'):
    seal=json.loads((AREA/name).read_text())
    for path,expected in seal['source_and_input_sha256'].items():assert sha(path)==expected,path
result=json.loads((AREA/'cpu-attempt1/result.json').read_text())
assert result['status']=='PASS' and all(all(row['comparisons'].values()) for row in result['records'])
source_map={str(path.relative_to(PACKAGE/'particlegan')):sha(path)
            for path in sorted((PACKAGE/'particlegan').rglob('*.py'))}
base_map={str(path.relative_to(BASE/'particlegan')):sha(path)
          for path in sorted((BASE/'particlegan').rglob('*.py'))}
assert set(source_map)==set(base_map)
assert [name for name in source_map if source_map[name]!=base_map[name]]==['feature_cells.py']
digest=hashlib.sha256()
for path in sorted((PACKAGE/'particlegan').rglob('*.py')):
    digest.update(str(path.relative_to(PACKAGE/'particlegan')).encode()+b'\0'+path.read_bytes()+b'\0')
patch=''.join(difflib.unified_diff((BASE/'particlegan/feature_cells.py').read_text().splitlines(keepends=True),
    (PACKAGE/'particlegan/feature_cells.py').read_text().splitlines(keepends=True),
    fromfile='a/particlegan/feature_cells.py',tofile='b/particlegan/feature_cells.py'))
# The first seal stopped after writing this exact patch, because it expected
# generic PASS instead of the independent receipt's specific PASS label.
assert (AREA/'RESOLUTION.patch').read_text()==patch
independent=ROOT/'performance/training-regression/count-review/post-ra8-quality/resolution-review/static-receipt.json'
review=json.loads(independent.read_text())
assert review['status']=='PASS_STATIC_SOURCE_AND_SCALAR_STATE'
numfiles=[*sorted((PACKAGE/'particlegan').rglob('*.py')),
          *[AREA/name for name in ('contract.py','prepare_source.py','prepare_tests.py','seal_owner.py','config.json')]]
numerical={str(path):sha(path) for path in numfiles}
source=json.loads((AREA/'SOURCE-FROZEN.json').read_text())['source_and_input_sha256']
extra=[AREA/name for name in ('SOURCE-FROZEN.json','TESTS-FROZEN.json','REPORT.md',
    'RESOLUTION.patch','cpu-attempt1/result.json','cpu-attempt1.log','seal-attempt1.json')]+[independent]
reviewable={**source,**{str(path):sha(path) for path in extra},**numerical}
receipt=dict(status='PASS',scope='finite even-fit average resolution; exact saved-input toy reactions and schema contracts, quality untested',
    cpu_result_sha256=sha(AREA/'cpu-attempt1/result.json'),independent_static_receipt_sha256=sha(independent),
    source_sha256=reviewable,numerical_source_sha256=numerical,source_guard='EXACT',cpu_only=True,process_exited=True,
    source_package_changed_modules=['feature_cells.py'],backend_schema=8,trainer_schema=5,quality_verdict=None,
    frozen_utc=datetime.now(timezone.utc).isoformat())
write('receipt.json',receipt)
ready=dict(status='FROZEN_CPU_QUALIFIED',package_root=str(PACKAGE),package_sha256=digest.hexdigest(),
    package_source_sha256=source_map,numerical_source_sha256=numerical,source_sha256=reviewable,
    backend_schema=8,trainer_schema=5,config_path=str(AREA/'config.json'),config_sha256=sha(AREA/'config.json'),
    config_changes=['birth_death_cells64->128'],resolution_policy='even_fit_average_rows_per_effective_rank_floor1_v1',
    actual_cell_formula='min(requested,max(1,even_fit_rows//max(1,effective_rank)))',
    cpu_receipt_sha256=sha(AREA/'receipt.json'),source_freeze_sha256=sha(AREA/'SOURCE-FROZEN.json'),
    tests_freeze_sha256=sha(AREA/'TESTS-FROZEN.json'),independent_static_receipt_sha256=sha(independent),
    patch_path=str(AREA/'RESOLUTION.patch'),patch_sha256=sha(AREA/'RESOLUTION.patch'),
    owner_frozen=str(AREA/'FROZEN.json'),quality_verdict=None,default_package_promoted=False,
    gpu_execution_owner='root only; NOT RUN',frozen_utc=datetime.now(timezone.utc).isoformat())
write('READY.json',ready)
paths=[path for path in sorted(AREA.rglob('*')) if path.is_file() and path.name!='FROZEN.json']
final=dict(status='FROZEN_POST_EXIT',owner_ready_sha256=sha(AREA/'READY.json'),receipt_sha256=sha(AREA/'receipt.json'),
    file_sha256={str(path):sha(path) for path in paths},read_only_source_sha256=source,
    package_sha256=digest.hexdigest(),quality_verdict=None,frozen_utc=datetime.now(timezone.utc).isoformat())
write('FROZEN.json',final)
print(json.dumps(dict(status=ready['status'],package_sha256=digest.hexdigest(),
    ready_sha256=sha(AREA/'READY.json'),frozen_sha256=sha(AREA/'FROZEN.json'),
    receipt_sha256=sha(AREA/'receipt.json'),FC=source_map['feature_cells.py'],config_sha256=sha(AREA/'config.json')),indent=2))
