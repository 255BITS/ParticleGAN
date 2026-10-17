import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
FIRST=HERE.parent/'grid-covariance'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE/'PREPARATION-FROZEN.json').exists()
paths=[HERE/'PROTOCOL.md',HERE/'probe.py',Path(__file__),FIRST/'FROZEN.json',FIRST/'result.json',
       ROOT/'pkg-CB64-RA8/particlegan/feature_cells.py',ROOT/'quality/ra8/READY.json',
       ROOT/'validation-cb64-ra8/screens/runs/grid100/final-state.pt']
prep=dict(status='FROZEN_BEFORE_MEASUREMENT',source_and_input_sha256={str(p):sha(p) for p in paths})
(HERE/'PREPARATION-FROZEN.json').write_text(json.dumps(prep,indent=2)+'\n')
print(json.dumps(dict(status=prep['status'],files=len(paths),sha256=sha(HERE/'PREPARATION-FROZEN.json'))))
