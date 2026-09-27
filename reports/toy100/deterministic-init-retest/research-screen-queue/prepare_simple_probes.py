#!/usr/bin/env python3
"""Seal original config/inline-penalty probes with the reviewed initializer bridge.

Standard library only. This prepares source, never imports a learner or trains.
Existing six-hook preparations and their index are never changed.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
TEMPLATE=HERE.parent/'research-mode-hold-preparation'
OUTPUT=HERE/'simple-probe-preparation'
CONTRACTS={
    '5cc4d7b1a5e68b8387233d83d3a4bab83c018e69ea6623e41032b690eb6bbeef': 'Original config-only probe; no learner hooks.',
    '6008a4b0baa69ab21500d6f71bc3b3cfd852860f186f85a7b1123d6db4b8ed3b': 'Inline GradientPenalty._phi smooth radial cap only.',
    '459b03b1b062ae5debb391512f2bde0bfbcd521950caaedadead18ab98108d9f': 'Inline GradientPenalty.penalty real-only pressure; original zero-weight fake graph retained.',
    'b68488e3901c78c63c8275b335c75ecc0c59a69c93380417023d416baab25419': 'Inline GradientPenalty.penalty real R1 plus fake cap only.',
    'e35c75d0d40175450e1f23f530a9086010a97195741d90f7096be7227d97b8f8': 'Inline GradientPenalty.penalty asymmetric real/fake caps only.',
}
OPTIMIZER='Original research probe calls ordinary Adam.step with foreach=False/fused=False; no eager state or counter injection, no capturable override. Default device remains original CUDA context. This is not a current public GANTrainer result.'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    if OUTPUT.exists():raise FileExistsError('Never overwrite sealed simple-probe preparations')
    rows=[r for r in read(HERE/'queue.json')['rows'] if r.get('probe_sha256') in CONTRACTS]
    assert len(rows)==11
    seal=read(TEMPLATE/'manifest.json')
    for rel,want in seal['files'].items():assert sha(TEMPLATE/rel)==want
    OUTPUT.mkdir();(OUTPUT/'prepared-bundles').mkdir()
    (OUTPUT/'.gitignore').write_text('prepared/\n__pycache__/\n')
    entries=[];archives=[]
    for row in rows:
        assert set(row['active_local_sources'])=={'config.json','probe.py'}
        assert row['configuration']['model']=='gan' and row['configuration']['prior_kind']=='particles'
        archive=Path(row['candidate_archive']['path']);assert sha(archive)==row['candidate_archive']['sha256']
        name=re.sub('[^a-z0-9]+','-',row['candidate'].lower()).strip('-')+'-'+hashlib.sha256(row['id'].encode()).hexdigest()[:8]
        target=OUTPUT/'prepared'/name;target.mkdir(parents=True)
        exclude={'README.md','source-plan.json','source-preflight.json','run_research_mode_hold.py'}
        for rel in seal['files']:
            if rel in exclude or rel.startswith('ka2-source/'):continue
            dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(TEMPLATE/rel,dest)
        with zipfile.ZipFile(archive) as source:
            assert len(source.namelist())==len(set(source.namelist())) and set(source.namelist())==set(row['candidate_files'])
            for rel in source.namelist():
                assert Path(rel).name==rel
                raw=source.read(rel);assert hashlib.sha256(raw).hexdigest()==row['candidate_files'][rel]
                dest=target/'candidate-source'/rel;dest.parent.mkdir(exist_ok=True);dest.write_bytes(raw)
        probe=(target/'candidate-source/probe.py').read_text();ast.parse(probe)
        assert 'checkpoint' not in row['active_local_sources']
        assert 'torch.zeros(' not in probe and "'capturable'" not in probe
        plan=read(TEMPLATE/'source-plan.json')
        runtime=Path(plan['historical_runtime'])
        for rel,want in plan['historical_runtime_files'].items():assert sha(runtime/rel)==want
        plan.update(candidate='RESEARCH-'+row['candidate']+'-new-init',source_candidate=row['source_directory'],candidate_files=row['candidate_files'],source_queue_row=row['id'],source_definition=row,probe_source=str(target/'candidate-source/probe.py'),probe_binding=CONTRACTS[row['probe_sha256']],runtime_binding=row['runtime_binding'],historical_runtime_receipts=row['historical_runtime_receipts'],group_template_manifest_sha256=sha(TEMPLATE/'manifest.json'),optimizer_state=OPTIMIZER)
        plan['probe_contract']=dict(kind='ORIGINAL_INLINE_PENALTY_PROBE',sha256=row['probe_sha256'],description=CONTRACTS[row['probe_sha256']],active_local_python=['probe.py'],prior_constructor_hook=False,network_constructor_hook=False,optimizer=OPTIMIZER,final_checkpoint='NOT_RETAINED_BY_ORIGINAL_PROBE; do not claim continuation/replay evidence',cpu_review='Must execute original probe setup on CPU and stop at initializer capture before any optimizer or forward; inline penalty setup must not be replaced by KA2 hooks.')
        plan['learner_preservation']='Original probe/config bytes, inline losses and ordinary Adam wrapper unchanged. Only constructor binding uses the public standalone initializer. No later mechanism/latent/response/checkpoint hooks injected.'
        plan['policy']=dict(evaluation_steps=1200,num_particles=12,z_dim=4,batch_size=128,seed=0,prior_init_std=.5,evaluation_latent_seed=9,evaluation_global_noise='402+completed_step',original_candidate_config=row['configuration'],note='Frozen tiny host resources and evaluation budget; all original declared_recipe/run_legacy schedule and loss semantics preserved.')
        dump(target/'source-plan.json',plan)
        worker=(TEMPLATE/'run_research_mode_hold.py').read_text()
        substitutions={"source=HERE/'ka2-source'":"source=HERE/'candidate-source'","candidate='RESEARCH-KA2-new-init'":"candidate=plan['candidate']","optimizer_state='Original research probe eager CUDA scalar counters; not native public lazy Adam'":"optimizer_state=plan['optimizer_state']",'Prepared exact research-KA2 screen':'Prepared exact original-inline-penalty research screen'}
        for old,new in substitutions.items():assert worker.count(old)==1;worker=worker.replace(old,new)
        ast.parse(worker);(target/'run_research_mode_hold.py').write_text(worker)
        dump(target/'source-preflight.json',dict(status='PASS_SOURCE_ONLY_CPU_REVIEW_REQUIRED',template_manifest_sha256=sha(TEMPLATE/'manifest.json'),worker_substitutions=substitutions,probe_contract=plan['probe_contract'],all_original_source_bytes_preserved=True,bridge_sha256=sha(target/'initialization_bridge.py'),no_training=True))
        (target/'README.md').write_text('# '+plan['candidate']+'\n\nPrepared original inline-penalty/config probe on the common frozen tiny host. Original source/config preserved. Public standalone initialization replaces fresh parameter values, with original constructor RNG consumption retained. No historical tensors loaded.\n\nThis is research-host evidence. Original scheduled policies remain scheduled; no old pass or public API qualification transfers. A separate matching CPU constructor/source proof is mandatory. The original probe does not save final optimizer/model state, so the screen does not establish continuation or replay. Runtime provenance: `'+row['runtime_binding']+'`.\n')
        files={str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
        dump(target/'manifest.json',dict(schema=1,status='PREPARED_REQUIRES_INDEPENDENT_CPU_REVIEW',files=files))
        entries.append(dict(queue_row=row['id'],candidate=plan['candidate'],directory=str(target),directory_relative=str(target.relative_to(OUTPUT)),manifest_sha256=sha(target/'manifest.json'),source_plan_sha256=sha(target/'source-plan.json'),bridge_sha256=sha(target/'initialization_bridge.py'),worker_sha256=sha(target/'run_research_mode_hold.py'),status='PREPARED_CPU_REVIEW_PENDING',required_review=str(OUTPUT/'reviews'/name/'cpu-constructor-proof.json'),required_review_relative='reviews/'+name+'/cpu-constructor-proof.json'))
        files['manifest.json']=sha(target/'manifest.json');zip_path=OUTPUT/'prepared-bundles'/(name+'.zip')
        with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as bundle:
            for rel in sorted(files):
                info=zipfile.ZipInfo(rel,(2026,9,27,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16;bundle.writestr(info,(target/rel).read_bytes())
        archives.append(dict(queue_row=row['id'],directory_relative=str(target.relative_to(OUTPUT)),archive=str(zip_path.relative_to(OUTPUT)),archive_sha256=sha(zip_path),manifest_sha256=sha(target/'manifest.json'),files=files))
    dump(OUTPUT/'prepared-index.json',dict(schema=1,status='PREPARED_NOT_AUTHORIZED',rows=entries))
    dump(OUTPUT/'prepared-bundles/index.json',dict(schema=1,rows=archives))
    shutil.copyfile(HERE/'restore_prepared.py',OUTPUT/'restore_prepared.py')
    dump(OUTPUT/'source-review.json',dict(status='SOURCE_PREPARED_REQUIRES_INDEPENDENT_REVIEW',full_probe_hashes=CONTRACTS,original_base_probe='5cc4d7b1a5e68b8387233d83d3a4bab83c018e69ea6623e41032b690eb6bbeef',all_variants_reviewed_as_complete_base_plus_exact_diff=True,template_manifest_sha256=sha(TEMPLATE/'manifest.json'),initializer_bridge_sha256=sha(TEMPLATE/'initialization_bridge.py'),preparer_sha256=sha(Path(__file__)),rows=[dict(queue_row=r['queue_row'],manifest_sha256=r['manifest_sha256']) for r in entries],training=False))
    print(json.dumps(dict(prepared=len(entries),index=str(OUTPUT/'prepared-index.json'),quality='NOT_RUN',cpu_review='REQUIRED')))

if __name__=='__main__':main()
