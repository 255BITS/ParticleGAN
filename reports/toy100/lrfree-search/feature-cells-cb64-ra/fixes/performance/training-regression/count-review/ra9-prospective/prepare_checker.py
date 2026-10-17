"""Derive only the declared RA9 source/schema/cell-cap audit adaptation."""
import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
BASE=ROOT/'performance/training-regression/count-review/ra8-prospective/audit_checkpoints.py'
ORIGINAL=ROOT/'integration/review/audit_learned.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n')
assert sha(BASE)=='2dbc740f3a81651173dd4b95a5601456953e542463893e15a313bbe36769b548'
assert sha(ORIGINAL)=='683acf472985084a911edd6763d63d411772563d7fcb71df4a547d2a502347ba'
READY=ROOT/'quality/ra9/READY.json'
LANE=ROOT/'validation-cb64-ra9'
assert sha(READY)=='ba558fff064f5e5613656a1c1550431091efc9989f4dd34f1d85a58d17d55990'
assert sha(LANE/'source-freeze.json')=='09e6e11fd7c862d7d98fca9b5209e93be971d034ab0c6553cbd9a18bdeb6bec2'
checked={str(READY):sha(READY),str(LANE/'source-freeze.json'):sha(LANE/'source-freeze.json')}
for record,key,base in ((read(READY),'numerical_source_sha256',None),
                        (read(LANE/'source-freeze.json'),'local_sources',LANE),
                        (read(LANE/'source-freeze.json'),'external_sources',None)):
    for name,digest in record[key].items():
        p=Path(name) if base is None else base/name
        assert sha(p)==digest,p
        checked[str(p)]=digest
assert not (HERE/'audit_checkpoints.py').exists()
text=BASE.read_text().replace('RA8','RA9').replace('ra8','ra9')
changes=[dict(kind='lane/variant labels',before='RA8/ra8',after='RA9/ra9')]
def replace(before,after,count=1):
    global text
    assert text.count(before)==count,(before,text.count(before))
    text=text.replace(before,after)
    changes.append(dict(before=before,after=after,occurrences=count))

replace('f2f117d650dbe8eca58c0313d07b661a25ad953bd6b3b30c99507510ca9e5949',sha(READY))
replace('1292ef86f6b16d8928923fda267fb9f1efc6d8ac746a1f091a01397e032a3cf3',sha(LANE/'source-freeze.json'))
replace('da29f10340ddd0cf4c8452e234496579699bd8ea7e1154da22ffeac18796065c',read(READY)['package_sha256'])
replace("from types import SimpleNamespace\n", "from types import SimpleNamespace\nfrom fractions import Fraction\n")
replace("assert ready['backend_schema']==7", "assert ready['backend_schema']==8")
replace('trainer_schema=5,backend_schema=7', 'trainer_schema=5,backend_schema=8')
replace('BACKEND_SCHEMA=7', 'BACKEND_SCHEMA=8')
replace("assert bd['backend']=='feature_cells' and bd['backend_schema']==7", "assert bd['backend']=='feature_cells' and bd['backend_schema']==8")
replace('old-backend6', 'old-backend7',3)
replace("bad['backend_schema']=6", "bad['backend_schema']=7")
replace("=={'lr','prior_lr_mult','d_lr_mult'}", "=={'lr','prior_lr_mult','d_lr_mult','birth_death_cells'}")
replace("assert (ROOT/'configs/overrides-CB64-RA9.json').read_bytes()==(ROOT/'configs/overrides-CB64-RA7.json').read_bytes()",
        "assert (ROOT/'configs/overrides-CB64-RA9.json').read_text()==(ROOT/'configs/overrides-CB64-RA8.json').read_text().replace('\"birth_death_cells\": 64','\"birth_death_cells\": 128')")
replace('08359ed4406148faf6915144ef4265d930829de53775eeafc4f17a11b5667b4c',read(READY)['config_sha256'])
replace("    assert settings['latent_kernel']=='bounded_local_dv12_lineage'",
        "    assert settings['cells']==cfg['recipe']['birth_death_cells']==128\n    assert settings['resolution_policy']==paired_methods['CELL_RESOLUTION_POLICY']\n    assert settings['latent_kernel']=='bounded_local_dv12_lineage'")
replace("k=settings['cells'];assert last['count_multiplicity']==3*k+2", "k=last['cells'];assert k==paired_methods['_fit_cell_count'](settings['cells'],(n+1)//2,last['metric_rank'])\n        assert last['count_multiplicity']==3*k+2")
replace('backend7_geometry_stamp_consistent', 'backend8_geometry_stamp_and_actual_cap_consistent')
replace("paired_methods=dict(math=math,Q=.05,torch=torch)\nexec(compile(ast.Module(body=definitions,type_ignores=[]),'<frozen-backend7-semantic-checks>','exec'),paired_methods)",
        "resolution_nodes=[deepcopy(n) for n in feature_tree.body if (isinstance(n,ast.FunctionDef) and n.name=='_fit_cell_count') or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CELL_RESOLUTION_POLICY' for t in n.targets))]\nassert len(resolution_nodes)==2\npaired_methods=dict(math=math,Q=.05,torch=torch,Fraction=Fraction)\nexec(compile(ast.Module(body=resolution_nodes+definitions,type_ignores=[]),'<frozen-backend8-semantic-checks>','exec'),paired_methods)")
assert 'backend_schema=7' not in text and "backend_schema']==7" not in text
ast.parse(text)
target=HERE/'audit_checkpoints.py'
with target.open('x') as f:f.write(text)
derivation=dict(status='FROZEN_ADAPTATION',base_path=str(BASE),base_sha256=sha(BASE),new_sha256=sha(target),
    substitutions=changes,scope='All original sealed endpoint population/graph/action checks, with declared backend8/request128/actual even-fit cap state validation; no training or quality changes.',
    original_artifact_auditor_path=str(ORIGINAL),original_artifact_auditor_sha256=sha(ORIGINAL),
    original_artifact_auditor_unchanged=True,original_toy_gate={'precision_min':.9,'coverage':25,'mass_tv_max':.1},
    model_forwards=0,gradients=0,optimizer_updates=0,new_emissions=0,new_seeds=0,cuda=False)
write(HERE/'DERIVATION.json',derivation)
files={str(p):sha(p) for p in (Path(__file__),target,HERE/'DERIVATION.json')}
write(HERE/'CHECKER-FROZEN.json',dict(status='FROZEN_CPU_ONLY_AUDITOR',files=files,
    original_checker={str(BASE):sha(BASE),str(ORIGINAL):sha(ORIGINAL)},reviewed_hashes=checked,
    prospective_ready_sha256=sha(READY),lane_freeze_sha256=sha(LANE/'source-freeze.json'),
    frozen_utc=datetime.now(timezone.utc).isoformat(),numerical_checkpoint_loads_before_seal=0))
print(json.dumps(dict(status='FROZEN_CPU_ONLY_AUDITOR',checker_sha256=sha(target),
    freeze_sha256=sha(HERE/'CHECKER-FROZEN.json'),reviewed_hashes=len(checked))))
