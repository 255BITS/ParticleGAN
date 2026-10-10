"""Freeze the prospective finite-fit policy before any numerical contract."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

AREA=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
files=[AREA/name for name in ('DESIGN.md','config.json','prepare_source.py')]
files+=sorted((AREA/'pkg-RESOLUTION/particlegan').rglob('*.py'))
files+=sorted((ROOT/'pkg-CB64-RA8/particlegan').rglob('*.py'))
files+=[ROOT/name for name in (
    'quality/ra8/READY.json','configs/overrides-CB64-RA8.json',
    'validation-cb64-ra8/source-freeze.json',
    'validation-cb64-ra8/learned/training/toy/CB64-RA8/checkpoint-1250.pt',
    'validation-cb64-ra8/learned/training/toy/CB64-RA8/checkpoint-2000.pt',
    'validation-cb64-ra8/learned/training/toy/CB64-RA8/config.json',
    'quality/ra6/integration-contract/check_reaction.py',
    'performance/sampler-regression/cpu-plan-review/post-ra7-quality/paired-average/contract_utils.py',
    'integration/review/training-regression/post-ra4-quality/measure_saved_utils.py')]
record=dict(status='FROZEN_BEFORE_NUMERICAL_CONTRACT',frozen_utc=datetime.now(timezone.utc).isoformat(),
            source_and_input_sha256={str(path):sha(path) for path in files},
            numerical_contract='NOT_RUN',config_changes=['birth_death_cells64->128'],
            only_modified_module='feature_cells.py',backend_schema=8,trainer_schema=5,
            policy='even_fit_average_rows_per_effective_rank_floor1_v1')
with (AREA/'SOURCE-FROZEN.json').open('x') as handle:
    handle.write(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'frozen_utc':record['frozen_utc'],
                  'file_count':len(record['source_and_input_sha256']),
                  'source_freeze_sha256':sha(AREA/'SOURCE-FROZEN.json')}))
