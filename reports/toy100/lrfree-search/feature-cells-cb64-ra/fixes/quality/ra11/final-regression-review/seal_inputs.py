"""Stdlib-only final raw-input seal, after all original jobs and watcher exit."""
import argparse
from common import *

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=HERE/'completed-inputs')
args=parser.parse_args()
assert args.output.resolve().is_relative_to(HERE) and not args.output.exists()
prep=read(HERE/'PREPARATION-FROZEN.json');verify(prep['files'])
ready=read(HERE/'CHECKER-READY.json');mapping=dict(prep['files'])
events=[json.loads(line) for line in (LANE/'run.log').read_text().splitlines() if line.strip()]
completed=[row for row in events if row.get('event')=='job_complete']
expected={f'learned-toy-{VARIANT}',f'learned-mnist-{VARIANT}',f'replay-{VARIANT}'}|{'screen-'+task for task in ready['tasks']}
assert len(completed)==19 and {row['name'] for row in completed}==expected
assert all(row['returncode']==0 and row['status'] in ('COMPLETE','PASS','FAIL') for row in completed)
assert not any(row.get('event')=='queue_aborted' for row in events)
for row in completed:pin(mapping,row['result'],row['result_sha256'])
through=read(LANE/'owned-slot-through-19.json');assert through['returncode']==0 and through['owned_numerical_parallelism']==1
launch=read(ROOT/'quality/results/RA11-regressions-launch.json')
driver=require_exited(launch['pid'],launch['startticks'])
start=read(WATCHER/'MONITOR-START.json');watcher=require_exited(start['pid'],start['startticks'])
summary=read(MONITOR/'summary.json')
assert summary['completed']==summary['total']==16 and summary['status'] in ('PASS','FAIL')
assert {row['task'] for row in summary['records']}==set(ready['tasks'])
assert all(row['canonical_fixture_validity']=='VALID' and row['acceptance_status']==row['primary_status'] in ('PASS','FAIL') for row in summary['records'])
manifest=read(MONITOR/'READ-ONLY-ARTIFACT-MANIFEST.json')
for name,item in manifest['inputs'].items():pin(mapping,LANE/name,item['sha256'])
for root in (LANE,MONITOR):
    for path in sorted(root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and not path.name.endswith('.lock'):pin(mapping,path)
pin(mapping,launch['log'])
pin(mapping,WATCHER/'monitor-process.log')
args.output.mkdir(parents=True)
value=dict(status='COMPLETED_ORIGINAL_RA11_INPUTS_FROZEN',utc=now(),validation=str(LANE),package_root=str(PACKAGE),
    preparation_sha256=sha(HERE/'PREPARATION-FROZEN.json'),source_and_input_sha256=mapping,
    completed_jobs=completed,total_jobs=19,total_screens=16,driver_exit_identity=driver,watcher_exit_identity=watcher,
    canonical_summary_sha256=sha(MONITOR/'summary.json'),canonical_manifest_sha256=sha(MONITOR/'READ-ONLY-ARTIFACT-MANIFEST.json'),
    Torch_imported=False,PT_loaded=0,artifact_interpretation=False,
    pending_root_GO=True,quality_FAILs_preserved=True)
write_new(args.output/'INPUTS-FROZEN.json',value)
print(json.dumps(dict(status=value['status'],sha256=sha(args.output/'INPUTS-FROZEN.json'),guards=len(mapping))))
