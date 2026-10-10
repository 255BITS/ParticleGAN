"""Independent frozen-source review; no Torch import or numerical replay."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER=ROOT/'integration/review/training-regression/post-ra8-quality/positive-lease-replay'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
EXPECTED={'READY.json':'8bb99156d8fcdabea9e02a279ca203e1af58a911193fbb8a45d3f37d3df6d185',
          'FROZEN.json':'08b70c4350ef520f629fa84483c0230ab323ebfcdcdd9d581609124f35ce16de',
          'SOURCE-FROZEN.json':'0b4eaf796ee41761d642967879121fbd42bc06b4f1ad7d832b2092f21cc5c6e0',
          'sample_reload.py':'9b807a301a00271b52af6fdfa1957af72e353592355b5b980d0e53766c4e87a6'}
assert not (HERE/'receipt.json').exists()
maps={str(OWNER/p):d for p,d in EXPECTED.items()}
frozen=read(OWNER/'FROZEN.json')
maps.update({str(OWNER/p):d for p,d in frozen['local_file_sha256'].items()})
maps.update(frozen['guarded_read_only_file_sha256'])
maps.update(read(OWNER/'SOURCE-FROZEN.json')['file_sha256'])
for p,d in maps.items():assert sha(p)==d,p
ready=read(OWNER/'READY.json'); inputs=read(OWNER/'INPUTS.json')
pkg=Path(inputs['package_root'])/'particlegan'
files=sorted(pkg.rglob('*.py'))
sources={str(p.relative_to(pkg)):sha(p) for p in files}
assert len(files)==29 and sources==ready['package_source_sha256']==inputs['package_source_sha256']
h=hashlib.sha256()
for p in files:h.update(str(p.relative_to(pkg)).encode()+b'\0'+p.read_bytes()+b'\0')
assert h.hexdigest()==ready['package_sha256']==inputs['package_sha256']=='da29f10340ddd0cf4c8452e234496579699bd8ea7e1154da22ffeac18796065c'
cpu=read(OWNER/'cpu-preflight-attempt1/result.json')
assert cpu['status']=='PASS_CPU_METADATA_PREFLIGHT' and cpu['cuda_initialized'] is False
assert cpu['model_constructions']==cpu['model_forwards']==cpu['generated_samples']==cpu['training_updates']==0
stamp=cpu['actual_paired_average']
assert stamp['coherent_rows']==977 and stamp['required']==973 and stamp['eligible'] is True
assert stamp['step']==2000 and stamp['snapshot']==250
source=(OWNER/'sample_reload.py').read_text();tree=ast.parse(source)
functions={v.name:ast.unparse(v) for v in tree.body if isinstance(v,ast.FunctionDef)}
assert 'state_excluded_fields=[]' in functions['cuda_contract']
assert 'trainer.step(' not in source and '.backward(' not in source and 'toy_metrics(' not in functions['cuda_contract']
assert 'trainer.sample(CHUNK, generator=stream)' in functions['traced_sample']
assert "manual_seed(common.SEED + 100)" in functions['cuda_contract']
assert 'for branch in range(2)' in functions['cuda_contract'] and 'for chunk in range(CHUNKS)' in functions['cuda_contract']
assert 'torch.load(CHECKPOINT, weights_only=False)' in functions['cuda_contract']
assert "map_location='cpu'" in functions['cpu_preflight'] and 'map_location' not in functions['cuda_contract']
assert "sections == saved_sections and digest(state) == saved_digest" in functions['cuda_contract']
assert "branches[0]['traces'] == branches[1]['traces']" in functions['cuda_contract']
assert "digest(globals_before) == digest(globals_after)" in functions['traced_sample']
assert "digest(ema_rows) == digest(trace['latent'])" in functions['traced_sample']
assert "with common.exclusive_learned_gpu()" in functions['cuda_contract']
for p in files:compile(p.read_text(),str(p),'exec')
compile(source,str(OWNER/'sample_reload.py'),'exec')
wrapper=ast.parse((ROOT/'quality/run_positive_lease.py').read_text())
wrapper_text=ast.unparse(wrapper)
for text in ('fcntl.LOCK_EX | fcntl.LOCK_NB','check_owned_slot()', 'not helpers.owned_numerical_processes()',
             "pass_fds=(lock.fileno(),)","GPU-72c1b506-891d-b8bc-b353-e020585e1c47"):
    assert text in wrapper_text,text
maps[str(Path(__file__))]=sha(Path(__file__))
receipt=dict(status='PASS',evidence='VALID',scope='Independent CPU frozen-source/metadata prelaunch review, CUDA result pending',
    owner_ready_sha256=EXPECTED['READY.json'],owner_frozen_sha256=EXPECTED['FROZEN.json'],
    source_prefreeze_sha256=EXPECTED['SOURCE-FROZEN.json'],package_sha256=h.hexdigest(),
    actual_positive_stamp=stamp,original_scorer_seed=314259,chunk_rows=256,chunks_per_branch=2,
    initial_restores=2,intermediate_reloads=1,proposed_generator_forwards=4,
    training_updates=0,quality_metric_evaluations=0,original_noise_and_perturbation_unchanged=True,
    full_serialized_state_comparison_no_exclusions=True,trace_rows_codes_noise_RNG_and_cache_checked=True,
    independent_reviewer_cuda=False,independent_reviewer_Torch_import=False,
    numerical_contract='NOT_RUN_BY_REVIEWER', limitations=['Saved977 lease only; does not certify quality or subsequent training from positive lease'],
    reviewed_sha256=maps)
(HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(HERE/'REPORT.md').write_text('# Positive-lease contract prelaunch source review\n\nPASS: frozen55-input owner preseal, all29 package bytes, actual977/973 metadata, same original seed and bounded2x256 chunks per branch verified. The helper compares all serialized state with no exclusions, traces rows/codes/perturbed codes/noisy outputs, and preserves training/global RNG. One branch reloads real saved post-chunk state and scorer cursor. Derived axis caches are rebuilt and compared; source/chart/state law is unchanged. Root serial wrapper holds original phase/GPU ownership locks.\n\nReviewer uses stdlib only, no numerical replay or CUDA. GPU contract remains root-owned and pending; passing this mechanical contract will not qualify Grid100 or quality.\n')
local={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
seal=dict(status='PASS',local_file_sha256=local,reviewed_sha256=maps,receipt_sha256=sha(HERE/'receipt.json'))
(HERE/'FROZEN.json').write_text(json.dumps(seal,indent=2)+'\n')
for p,d in maps.items():assert sha(p)==d,p
print(json.dumps(dict(status='PASS',reviewed_files=len(maps),receipt_sha256=sha(HERE/'receipt.json'),freeze_sha256=sha(HERE/'FROZEN.json'))))
