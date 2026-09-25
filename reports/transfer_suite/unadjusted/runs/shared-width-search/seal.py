"""Copy this completed evidence bundle without altering raw/source bytes."""
from datetime import datetime,timezone
import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/tmp/pr38-shared-width-search')
DEST=Path.cwd()/'reports/transfer_suite/unadjusted/runs/shared-width-search'
audit=json.loads((ROOT/'audit.json').read_text())
assert audit['episodes']==35 and audit['width_winners']==['width_last_softplus8_128_l3']
assert not DEST.exists()
indexes=[f'reports/transfer_suite/unadjusted/runs/shared-width-search/{phase}/index.json' for phase in ('screen','refinement','last_refinement','cross')]
(ROOT/'import.json').write_text(json.dumps({'append_to_existing_entry':'shared_c6','indexes':indexes,'raw_width_trials':34,'rare_cross_checks':1,'generated_aggregate_untouched':True},indent=2)+'\n')
shutil.copytree(ROOT,DEST)
files={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='archive_manifest.json'}
(DEST/'archive_manifest.json').write_text(json.dumps({'version':'shared-width-search-v3','sealed_utc':datetime.now(timezone.utc).isoformat(),'files':files,'original_byte_hashes_preserved':True,'episodes':35},indent=2,sort_keys=True)+'\n')
for name,want in files.items():assert hashlib.sha256((DEST/name).read_bytes()).hexdigest()==want
print(f'Sealed {len(files)} files at {DEST}')
