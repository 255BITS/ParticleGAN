"""Finalize immutable owner/root resolution source qualification; no numerical rerun."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra8-quality/resolution'
OUT=ROOT/'quality/ra9/independent-source'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
EXPECTED={'READY.json':'f5afd255877d747a6113881aedffdcb8a62d01e01a3c4f7b486e4e1da62081af',
          'FROZEN.json':'fa9aee80a4d62fe6d611e69508caaa66ab909dd6bf176d918cee6e5581a47a40'}
assert not (OUT/'receipt.json').exists()
for name,d in EXPECTED.items():assert sha(OWNER/name)==d,name
owner=read(OWNER/'READY.json');seal=read(OWNER/'FROZEN.json')
static=read(HERE/'static-receipt.json');static_seal=read(HERE/'STATIC-FROZEN.json')
assert owner['status']=='FROZEN_CPU_QUALIFIED' and seal['status']=='FROZEN_POST_EXIT'
assert static['status']=='PASS_STATIC_SOURCE_AND_SCALAR_STATE'
assert seal['owner_ready_sha256']==sha(OWNER/'READY.json')
maps={}
def add(mapping):
    for p,d in mapping.items():
        assert p not in maps or maps[p]==d,('conflicting map',p)
        maps[p]=d
add(static['reviewed_sha256']);add(seal['file_sha256']);add(seal['read_only_source_sha256'])
add(owner['numerical_source_sha256'])
for p,d in static_seal['local_file_sha256'].items():add({str(HERE/p):d})
composition=ROOT/'quality/ra9/COMPOSITION.json'
assert sha(composition)=='9e4766abe810c857e7790ca55277b62550ae2ff8ab690185b58373223a563d59'
composed=read(composition);add(composed['composed_from'])
pkg=ROOT/'pkg-CB64-RA9/particlegan';proposal=Path(owner['package_root'])/'particlegan'
files=sorted(pkg.rglob('*.py'))
sources={str(p.relative_to(pkg)):sha(p) for p in files}
assert len(files)==29 and sources==composed['source_sha256']==owner['package_source_sha256']
for p in files:
    name=str(p.relative_to(pkg));assert p.read_bytes()==(proposal/name).read_bytes()
    if name!='feature_cells.py':assert p.read_bytes()==(ROOT/'pkg-CB64-RA8/particlegan'/name).read_bytes()
    add({str(p):sources[name]});compile(p.read_text(),str(p),'exec')
h=hashlib.sha256()
for p in files:h.update(str(p.relative_to(pkg)).encode()+b'\0'+p.read_bytes()+b'\0')
assert h.hexdigest()==owner['package_sha256']==composed['package_sha256']=='2d00c77ae0e7545ac253ff81aa729158d69016be82edc117e5597bdd664b86ce'
assert sources['feature_cells.py']==static['source_sha256']=='39558fb3839090eb9d933b8b37e4c73b7dfc3cf123b24c2ef58d818cc4052fcc'
config=ROOT/'configs/overrides-CB64-RA9.json';oldconfig=ROOT/'configs/overrides-CB64-RA8.json'
assert sha(config)==composed['config_sha256']=='b3656ea7494413106484c556e53877900dd5ccebf2597b3f80b7d0e7d5ba4437'
original=oldconfig.read_text();assert original.count('"birth_death_cells": 64')==1
assert config.read_text()==original.replace('"birth_death_cells": 64','"birth_death_cells": 128')
assert read(config)==read(OWNER/'config.json')
assert owner['backend_schema']==composed['backend_schema']==8 and owner['trainer_schema']==composed['trainer_schema']==5
assert owner['resolution_policy']==static['policy']
cpu=read(OWNER/'cpu-attempt1/result.json')
assert sha(OWNER/'cpu-attempt1/result.json')==owner['cpu_receipt_sha256']==seal['receipt_sha256']
assert cpu['status']=='PASS' and cpu['cpu_only'] is True and cpu['cuda_initialized'] is False
assert cpu['new_training_steps']==cpu['new_optimizer_steps']==cpu['new_quality_emissions']==cpu['new_seeds']==0
assert cpu['quality_verdict'] is None and cpu['source_input_tensors_and_global_rng_unchanged'] is True
assert cpu['source_freeze_sha256']==owner['source_freeze_sha256']==sha(OWNER/'SOURCE-FROZEN.json')
assert cpu['tests_freeze_sha256']==owner['tests_freeze_sha256']==sha(OWNER/'TESTS-FROZEN.json')
assert len(cpu['records'])==2 and [r['step'] for r in cpu['records']]==[1250,2000]
for r in cpu['records']:
    assert all(r['comparisons'].values()) and all(r['baseline_checks'].values()) and all(r['proposal_checks'].values())
    assert r['actual_cells']==64 and r['requested_baseline']==64 and r['requested_proposal']==128
    assert r['ordinary']==51 and r['ordinary']==r['copies']+r['novel']
assert len(cpu['metadata']['atomic_invalid_controls'])==13 and cpu['metadata']['fresh_roundtrip'] is True
assert cpu['metadata']['odd_ceil_typed_control']['fitted_rows']==401
assert cpu['semantic_exclusions']==['backend_schema7->8','settings.cells64->128','new settings.resolution_policy','birth_death.last.eval_seconds']
for p in (OWNER/'READY.json',OWNER/'FROZEN.json',HERE/'STATIC-FROZEN.json',HERE/'static-receipt.json',
          composition,config,oldconfig,Path(__file__)):
    add({str(p):sha(p)})
for p,d in maps.items():assert sha(p)==d,p
receipt=dict(status='PASS',evidence='VALID',scope='Independent final root source/scalar-state qualification; owner numeric contracts reviewed, no reviewer numerical rerun',
    package_root=str(pkg.parent),package_sha256=h.hexdigest(),source_sha256=sources,config_sha256=sha(config),
    owner_READY_sha256=EXPECTED['READY.json'],owner_FROZEN_sha256=EXPECTED['FROZEN.json'],
    composition_sha256=sha(composition),backend_schema=8,trainer_schema=5,
    resolution_policy=static['policy'],actual_cell_formula=owner['actual_cell_formula'],
    config_changes={'birth_death_cells':{'before':64,'after':128}},
    all29_root_modules_byte_exact_owner=True,other28_modules_byte_exact_RA8=True,
    exact_RA8_config_text_single_replacement=True,whole_FC_AST_inverse_RA8=True,
    even_only_cap_no_odd_fake_or_benchmark_inputs=True,
    scalar_state=static['accepted_scalar_states'],malformed_metadata_rejected=static['rejected_scalar_metadata'],
    old_backend7_actual_first_guard_rejection_without_mutation=True,
    actual_family='K+2K+2 at Q/(3K+2), including empty categories',
    toy512_rank8_actual64_family194=True,grid10000_rank8_actual128_family386=True,
    owner_closed_CPU_contract=dict(receipt_sha256=sha(OWNER/'cpu-attempt1/result.json'),records=cpu['records'],
        edge_cases=cpu['edges'],metadata=cpu['metadata'],fixture_scope=cpu['fixture_scope'],
        semantic_exclusions=cpu['semantic_exclusions']),
    unchanged_training_noise_serving_population_quotas_and_parent_reservations=True,
    reviewer_Torch_import=False,reviewer_model_forwards=0,reviewer_draws=0,reviewer_training_updates=0,
    quality_verdict=None,prospective_target='Original final toy AND full7000 Grid100 must pass; candidate not yet tested',
    limitations=['Average fitted-reference resolution cap, no per-cell occupancy/quality guarantee',
        'Finer K increases count multiplicity and O(NK) chart costs; conditional-iid/adaptive limitations unchanged',
        'Full actual trainer API is separate integration-contract scope; no production checkpoint migration'],
    private_failed_inspection='Source hash changed before scalar inspection; retained failed-inspection1.txt',reviewed_sha256=maps)
report='# RA9 independent source and state qualification\n\nPASS/VALID. All29 composed modules equal the frozen owner, with only feature_cells.py changed from RA8. Full AST inversion restores RA8 after the even-fit cap, schema8/policy and load-validation splices. Root config bytes are exactly RA8 with requested cells64→128; no other config value changes.\n\nEffective K uses even rows and effective rank only, with rank0 denominator1 and disabled serving. Scalar odd/initial/degenerate/actual-cap controls pass; malformed typed partition/cell/family/cutoff controls reject. Actual first backend guard rejects old7 state before mutation; trainer5 is retained and byte unchanged. Owner closed CPU1250/2000 reaction contracts preserve both plans/count laws/numerical state/RNG at actual64, with explicit private fixture metadata rebinding. No reviewer numerical rerun, Torch import, forward, draws or updates.\n\nThe cap controls average resolution, not each cell occupancy. Actual3K+2 family grows194→386 on the larger chart, reducing local power and increasing O(NK) cost. Existing conditional-iid/adaptive limits remain. No quality claim or checkpoint migration: both unchanged original final toy and full7000 Grid100 are still required. Full actual trainer API is the separate integration-contract review.\n'
OUT.mkdir(parents=True,exist_ok=True)
for path,value in ((HERE/'receipt.json',receipt),(OUT/'receipt.json',receipt)):
    with path.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
for path in (HERE/'REPORT.md',OUT/'REPORT.md'):
    with path.open('x') as f:f.write(report)
print(json.dumps(dict(status='PASS',reviewed_files=len(maps),receipt_sha256=sha(OUT/'receipt.json'),root_output=str(OUT))))
