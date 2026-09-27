#!/usr/bin/env python3
"""Prepare source-only DV2/DV3 original recovery-ring retests; never import Torch."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'continuous-api-search/supervisor-support/k3p-ring-reference'
EVIDENCE=HERE.parent/'continuous-api-search/evidence'
OUTPUT=HERE/'dv23-single-shift-preparation'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def replace(source,old,new):
    assert source.count(old)==1,(old[:120],source.count(old))
    return source.replace(old,new)
def spans(source):
    return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

PREFLIGHT='''"""Standard-library source verification only."""
from pathlib import Path
import ast
import hashlib
import json
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(bundle):
    manifest=json.loads((bundle/'bundle-sha256.json').read_text())
    declaration=json.loads((bundle/'declaration.json').read_text())
    for name,wanted in manifest['files'].items():
        assert sha(bundle/name)==wanted, name
    package={str(p.relative_to(bundle/'source')):sha(p) for p in (bundle/'source/particlegan').glob('*.py')}
    assert package==declaration['package_sha256']
    recipe=declaration['recipe']
    assert recipe['initialization']=='batch_feature_zero' and recipe['total_steps'] is None
    assert (recipe['num_particles'],recipe['z_dim'],recipe['batch_size'])==(20000,2,2048)
    assert recipe['input_noise_std']==0 and recipe['output_noise_std']==.029 and recipe['output_noise_warmup']==0
    assert recipe['continuous_policy'] in ('dv2','dv3')
    for name in ('worker.py','ring_contract.py','execution_contract.py','source/frozen_host.py'):ast.parse((bundle/name).read_text())
    return dict(status='PASS_SOURCE_ONLY',manifest_sha256=sha(bundle/'bundle-sha256.json'),package_files=len(package),quality='NOT_RUN')
if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent),indent=2))
'''

CONTRACT='''"""Frozen public ring construction. Imported code performs no construction."""
import hashlib

def construct(torch,package,host,recipe,device):
    assert str(torch.get_default_device())=='cpu'
    generator=host.SimpleMLPGenerator(recipe.z_dim,96,3,2).to(device)
    critic=host.SimpleMLPDiscriminator(2,96,3,3).to(device)
    return package.GANTrainer(recipe,generator,critic,seed=0,
        optimizer_options={'foreach':False,'fused':False})

def material(torch,trainer):
    def visit(value):
        if isinstance(value,torch.Tensor):
            t=value.detach().cpu().contiguous()
            return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        if isinstance(value,dict):return {str(k):visit(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [visit(v) for v in value]
        if value is None or isinstance(value,(str,int,float,bool)):return value
        raise TypeError(type(value))
    state=trainer.state_dict()
    for key in ('streams','cpu_rng','cuda_rng','device'):state.pop(key,None)
    modules={}
    for role in ('G','D','prior','ema_G','ema_D','ema_prior'):
        m=getattr(trainer,role)
        modules[role]=dict(parameters=dict(m.named_parameters()),buffers=dict(m.named_buffers()))
    return visit(dict(state=state,all_parameters_and_buffers=modules))
'''

def main():
    if OUTPUT.exists():raise FileExistsError('Do not overwrite a sealed preparation')
    OUTPUT.mkdir();(OUTPUT/'originals').mkdir()
    for name in ('worker.py','execution_contract.py','declaration.json','source/frozen_host.py'):
        dest=OUTPUT/'originals'/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(REFERENCE/name,dest)
    reference=read(REFERENCE/'declaration.json')
    worker=(REFERENCE/'worker.py').read_text()
    worker=replace(worker,'Sealed public K3P v0.8.0 recovery-ring reference.','Sealed deterministic-init DV2/DV3 original recovery-ring retest.')
    worker=replace(worker,'    parser.add_argument("--output", type=Path, required=True)','    parser.add_argument("--output", type=Path, required=True)\n    parser.add_argument("--reviewed-cpu-proof", type=Path, required=True)')
    worker=replace(worker,'    args.output.mkdir(parents=True, exist_ok=False)','''    proof = json.loads(args.reviewed_cpu_proof.read_text())
    assert proof['status']=='PASS' and proof['cuda_initialized'] is False and proof['learner_steps']==0
    assert proof['manifest_sha256']==hashlib.sha256((BUNDLE/'bundle-sha256.json').read_bytes()).hexdigest()
    assert proof['declaration_sha256']==hashlib.sha256((BUNDLE/'declaration.json').read_bytes()).hexdigest()
    assert all(proof['checks'].get(k) is True for k in ('all_initial_tensors_repeat_without_seed_reset','initializer_rng_neutral','original_constructor_order','original_prior_std1','all_buffers_and_checkpoint_state_captured'))
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output/'reviewed-cpu-proof.json',proof)''')
    worker=replace(worker,'        from particlegan.recipes import learning_rate_scales','        import particlegan as package\n        from ring_contract import construct, material')
    worker=replace(worker,'declaration["release"]["source_sha256"][relative]','declaration["package_sha256"][relative]')
    worker=replace(worker,'            assert not any(n.startswith(("benchmarks.", "particlegan.ka2", "particlegan.precision", "particlegan.game_update")) for n in sys.modules)','            assert not any(n.startswith("benchmarks.") for n in sys.modules)')
    worker=replace(worker,'        from particlegan import GANTrainer, get_recipe','        from particlegan import GANTrainer, Recipe')
    worker=replace(worker,'        recipe = get_recipe(total_steps=4600)','        recipe = Recipe(**declaration["recipe"])')
    worker=replace(worker,'        assert input_noise_std(recipe, 460) == 0 and output_noise_std(recipe, 920) == .029','        assert recipe.total_steps is None\n        assert input_noise_std(recipe, 0) == 0 and output_noise_std(recipe, 0) == .029')
    worker=replace(worker,'''            generator = host.SimpleMLPGenerator(recipe.z_dim, 96, 3, 2).to("cuda:0")
            critic = host.SimpleMLPDiscriminator(2, 96, 3, 3).to("cuda:0")
            value = GANTrainer(recipe, generator, critic, seed=0,
                               optimizer_options={"foreach": False, "fused": False})''','''            value = construct(torch,package,host,recipe,"cuda:0")''')
    worker=replace(worker,'        trainer = make_trainer()','''        trainer = make_trainer()
        actual_material=material(torch,trainer)
        dump(args.output/'initial-material.json',actual_material)
        assert actual_material==proof['all_initial_material'], 'actual CUDA initialization differs from independent CPU proof'
        dump(args.output/'initial-model-cpu-cuda-proof.json',dict(status='PASS',all_initial_non_rng_state_and_named_parameters_and_buffers_equal=True))''')
    worker=replace(worker,'''                network, prior_scale = learning_rate_scales(step - 1, recipe)
                for optimizer, base_rates, roles in zip((trainer.opt_g,trainer.opt_d),trainer.initial_lrs,trainer.roles):
                    for group,rate,role in zip(optimizer.param_groups,base_rates,roles):
                        assert group["lr"] == rate * (prior_scale if role == "prior" else network)
''','')
    worker=replace(worker,'                                accepted_updates=step, field_evaluations=1)','                                accepted_updates=step, policy=trainer.controller.diagnostics())')
    worker=replace(worker,'                                 learning_rates=host.rates(trainer), penalty=trainer.penalty.diagnostics(),','                                 learning_rates=host.rates(trainer), penalty=trainer.penalty.diagnostics(), policy=trainer.controller.diagnostics(),')
    ast.parse(worker)
    (OUTPUT/'worker-changes.patch').write_text(''.join(difflib.unified_diff((REFERENCE/'worker.py').read_text().splitlines(True),worker.splitlines(True),fromfile='reviewed-k3p-ring-reference/worker.py',tofile='dv23-newinit-ring/worker.py')))
    entries=[]
    for name in ('api-dv2','api-dv3'):
        old=EVIDENCE/(name+'-single');port=HERE/'port-source'/name
        old_decl=read(old/'declaration.json');new_port=read(port/'candidate-declaration.json')
        assert new_port['serial_backward_argument'] is False and new_port['initial_optimizer_state']=='native_lazy'
        target=OUTPUT/name;target.mkdir();(target/'source').mkdir()
        (target/'worker.py').write_text(worker);(target/'ring_contract.py').write_text(CONTRACT);(target/'preflight.py').write_text(PREFLIGHT)
        execution=(REFERENCE/'execution_contract.py').read_text().replace('get("schema") != 3','get("schema") != 4').replace('requires unchanged released trainer schema3','requires exact DV2/DV3 trainer schema4')
        (target/'execution_contract.py').write_text(execution)
        shutil.copyfile(REFERENCE/'source/frozen_host.py',target/'source/frozen_host.py')
        shutil.copyfile(old/'source.zip',OUTPUT/'originals'/(name+'-source.zip'))
        shutil.copyfile(old/'declaration.json',OUTPUT/'originals'/(name+'-declaration.json'))
        with zipfile.ZipFile(port/'package.zip') as z:
            assert {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}==new_port['package_sha256']
            for rel in z.namelist():
                assert rel.startswith('particlegan/') and '..' not in Path(rel).parts
                path=target/'source'/rel;path.parent.mkdir(exist_ok=True);path.write_bytes(z.read(rel))
        host=(target/'source/frozen_host.py').read_text();selected=spans(host);host_receipts={}
        with zipfile.ZipFile(old/'source.zip') as z:
            for source,names in {'benchmarks/locked_shared/mlp.py':['SimpleMLPGenerator','SimpleMLPDiscriminator'],'benchmarks/locked_shared/mode_hold.py':['ring_means','diversity'],'reports/data-drift-api/worker.py':['digest','rates','state_receipt']}.items():
                original=spans(z.read(source).decode())
                for symbol in names:
                    assert selected[symbol]==original[symbol]
                    host_receipts[symbol]=dict(original_source=source,sha256=hashlib.sha256(original[symbol].encode()).hexdigest())
        recipe=old_decl['recipe']|{'initialization':'batch_feature_zero'}
        old_initial=read(old/'initial.json')
        initial_keys=('streams_sha256','cpu_rng_sha256','cuda_rng_sha256','real_stream_sha256','means_sha256')
        declaration=dict(schema=1,status='PREPARED_REQUIRES_INDEPENDENT_CPU_SOURCE_REVIEW',candidate=name.upper()+'-new-init',gate='single_shift4600_new_init',initializer_commit=new_port['initializer_commit'],package_sha256=new_port['package_sha256'],port_manifest_sha256=sha(port/'port-manifest.json'),port_package_sha256=sha(port/'package.zip'),historical_source_zip_sha256=sha(old/'source.zip'),historical_declaration_sha256=sha(old/'declaration.json'),recipe=recipe,host=reference['host']|{'initialization':'torch.manual_seed(0); G CPU constructor then CUDA; D CPU constructor then CUDA; exact candidate GANTrainer constructs prior on CPU at std1 then moves it to CUDA. Public batch_feature_zero initializes fresh G/D/prior without consuming RNG.'},evaluation=reference['evaluation']|{'frozen_control':'At2400 save full main state, construct separate identical candidate trainer, load saved schema4 state restoring construction-changed global RNG; no optimizer updates after copy. Never reload main learner.'},execution=reference['execution'],runtime_expected=reference['runtime_expected'],expected_initial_fixture={k:old_initial[k] for k in initial_keys},initial_comparison_scope='Compare original ring RNG/means only; all tensors freshly use new initializer. Every initialized state tensor and named parameter/buffer must also equal independent CPU proof.',host_source_receipt=host_receipts,protocol=old_decl['protocol']['protocols']['single_shift'],mechanism=old_decl['mechanism'],quality_policy='Retain all460 observations, acquisition/retention to2400 and shifted arrival/stability to4600. No81 deadline, seed variant, initializer retry, policy repair, or old pass inheritance.',construction='G CPU constructor thenCUDA; D CPU constructor thenCUDA; Trainer-owned prior CPU factory at original default std1 from global CPU RNG after network construction; separate latent seed2, penalty3, eval4 and noise5 streams; public initializer RNG-neutral.',checkpoint='Schema4 trainer/controller/optimizers/EMA/RNG plus caller stream+means and explicit full-step serial execution; freeze at2400 without an update. No midrun reset of main learner.',old_quality_inherited=False)
        dump(target/'declaration.json',declaration)
        files={str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
        dump(target/'bundle-sha256.json',dict(schema=1,files=files))
        entries.append(dict(candidate=declaration['candidate'],directory=str(target),manifest_sha256=sha(target/'bundle-sha256.json'),required_review=str(OUTPUT/'reviews'/name/'cpu-constructor-proof.json'),quality='NOT_RUN',source_status='PREPARED_REQUIRES_INDEPENDENT_CPU_REVIEW'))
    dump(OUTPUT/'prepared-index.json',dict(rows=entries,preparer_sha256=sha(Path(__file__)),reference_manifest_sha256=sha(REFERENCE/'bundle-sha256.json')))
    print(json.dumps(dict(prepared=2,index=str(OUTPUT/'prepared-index.json'),quality='NOT_RUN')))

if __name__=='__main__':main()
