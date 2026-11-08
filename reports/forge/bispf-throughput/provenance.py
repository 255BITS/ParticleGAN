"""Rebuild compact original profiling, software and publication provenance; no training."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,file_hash,read_json
from experiments.forge.trainer_families import scientific_row_hash
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts',type=Path,required=True)
args=parser.parse_args()
root=ROOT;art=args.artifacts.resolve();dest=root/'reports/forge/bispf-throughput'
profiles=[]
for f in (art/'profile').glob('*/summary.json'):
 row=read_json(f);profiles.append({**row,'original_summary':{'path':str(f),'sha256':file_hash(f)}})
atomic_json(dest/'profiling.json',dict(schema='bispf_throughput_component_readout_v1',profiles=profiles,declaration=dict(path=str(art/'profile/declaration.json'),sha256=file_hash(art/'profile/declaration.json'))))
probes=[]
for name in ['bf16-probe','bf16-channels-last-probe','bf16-graph-probe']:
 directory=art/name
 for f in directory.glob('*/result.json'):
  row=read_json(f)
  probes.append(dict(probe=name,task=f.parent.name,status=row['status'],metrics=row['metrics'],recipe=row['recipe'],identity=row['identity'],original_receipt=dict(path=str(f),sha256=file_hash(f)),qualification_input=False,
                     declaration_matches_execution=name!='bf16-probe',note=('First whole-update feasibility declaration survived a preflight rejection, while the later implementation scopes autocast to forwards. Retain as an unqualified development observation; the final registered experiments have separate correct bindings.' if name=='bf16-probe' else 'Development timing only; final source-frozen matrix is the comparison evidence.')))
atomic_json(dest/'probes.json',dict(schema='bispf_throughput_probes_v1',rows=probes,qualification_input=False,first_prototype_patch=dict(path=str(art/'bf16-probe/prototype.patch'),sha256=file_hash(art/'bf16-probe/prototype.patch'))))
before=json.loads(subprocess.check_output(['git','show','bbd95834c:reports/forge/technique-inventory.json'],text=True))
after=read_json(root/'reports/forge/technique-inventory.json')
proof=dict(schema='bispf_throughput_publication_integrity_v1',scientific_rows_unchanged=[scientific_row_hash(r) for r in before['rows']]==[scientific_row_hash(r) for r in after['rows']],selection_file_unchanged=subprocess.check_output(['git','show','bbd95834c:configs/forge/selections/family-current-v1.json'])==(root/'configs/forge/selections/family-current-v1.json').read_bytes(),public_default_unchanged=next(line for line in subprocess.check_output(['git','show','bbd95834c:particlegan/recipes.py'],text=True).splitlines() if line.startswith('DEFAULT_RECIPE ='))==next(line for line in (root/'particlegan/recipes.py').read_text().splitlines() if line.startswith('DEFAULT_RECIPE =')),refresh_receipt=read_json(art/'refresh-publication-scoped.log'))
assert proof['scientific_rows_unchanged'] and proof['selection_file_unchanged']
atomic_json(dest/'publication-integrity.json',proof)
checks=[]
for f,count in [('tests-initial.log',109),('tests-broad.log',209),('tests-cache-layout-final.log',14)]:
 log=art/f
 assert f'{count} passed' in log.read_text()
 checks.append(dict(path=str(log),sha256=file_hash(log),passed=count))
atomic_json(dest/'software-checks.json',dict(schema='bispf_throughput_software_checks_v1',checks=checks,layout_test=dict(original_failed_log=dict(path=str(art/'tests-cache-layout.log'),sha256=file_hash(art/'tests-cache-layout.log')),cause='Canonical contiguous graph inputs can change the reduction rounding of strided eager kernels. The corrected contract requires bitwise replay of the same contiguous calculation plus numerical agreement with strided eager output; no implementation or trained gate was changed.',final_passes=14),qualification_input=False))

state=read_json(art/'performance-queue/queue/state.json')
request=next(iter(state['submissions'].values()))['request']
require_source={name: file_hash(root/name)==digest for name,digest in request['source']['files'].items()}
assert all(require_source.values()), 'Current implementation differs from measured source'
atomic_json(dest/'source-integrity.json',dict(schema='bispf_throughput_source_integrity_v1',source_digest=request['source']['digest'],origin_commit=request['source']['origin_commit'],all_measured_source_files_match_current_checkout=True,verified_files=len(require_source),original_source_map=dict(path=str(art/'performance-queue/snapshots'/request['source']['digest']/'forge-source.json'),sha256=file_hash(art/'performance-queue/snapshots'/request['source']['digest']/'forge-source.json'))))
