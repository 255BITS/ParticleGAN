"""Seal focused helper before its numerical contracts; no Torch import."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
AREA=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
source=json.loads((AREA/'SOURCE-FROZEN.json').read_text())
for name,expected in source['source_and_input_sha256'].items():assert sha(Path(name))==expected,name
files=[AREA/name for name in ('contract.py','prepare_tests.py','SOURCE-FROZEN.json')]
record=dict(status='FROZEN_BEFORE_NUMERICAL_HELPER_EXECUTION',frozen_utc=datetime.now(timezone.utc).isoformat(),
            source_and_input_sha256={str(path):sha(path) for path in files},
            input_selection=[1250,2000],numerical_tests='NOT_RUN',cpu_only=True)
with (AREA/'TESTS-FROZEN.json').open('x') as handle:handle.write(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],tests_freeze_sha256=sha(AREA/'TESTS-FROZEN.json'))))
