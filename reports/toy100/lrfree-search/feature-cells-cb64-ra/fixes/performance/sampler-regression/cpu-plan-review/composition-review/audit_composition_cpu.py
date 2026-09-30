"""Independent read-only RA4 composition, API and semantic checkpoint audit."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
AXIS=HERE.parent/'pkg-AXIS-ID'
COUNT=ROOT/'integration/review/training-regression/global-count/pkg-global-count'
PERF=HERE.parent/'plan-batching/pkg-PLAN-FINAL'
PERF_READY=PERF.parent/'READY.json'
COMPOSE=ROOT/'compose4.py'
PROOF=ROOT/'integration/iteration-4/COMPOSITION.json'
HARNESS=Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(package):return {str(p.relative_to(package)):sha(p) for p in sorted((package/'particlegan').rglob('*.py'))}


def ast_hash(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def find(tree,name,owner=None):
    body=tree.body if owner is None else find(tree,owner).body
    return next(n for n in body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)


def assignment(method,target):
    return next(n for n in ast.walk(method) if isinstance(n,ast.Assign) and any(
        (isinstance(t,ast.Attribute) and t.attr==target) or (isinstance(t,ast.Name) and t.id==target)
        for t in n.targets))


def replace_node(tree,old,new):
    class Replace(ast.NodeTransformer):
        def visit(self,node):
            if node is old:return deepcopy(new)
            return super().visit(node)
    return Replace().visit(tree)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def static_checks(package):
    axis=ast.parse((AXIS/'particlegan/feature_cells.py').read_text())
    count=ast.parse((COUNT/'particlegan/feature_cells.py').read_text())
    perf=ast.parse((PERF/'particlegan/feature_cells.py').read_text())
    composed=ast.parse((package/'particlegan/feature_cells.py').read_text())
    ready=json.loads(PERF_READY.read_text())
    proof=json.loads(PROOF.read_text())
    assert proof['performance_ready_sha256']==sha(PERF_READY)
    assert proof['package_source_sha256']==sources(package)
    axis_map,candidate_map=sources(AXIS),sources(package)
    assert axis_map.keys()==candidate_map.keys()
    assert {name for name in axis_map if axis_map[name]!=candidate_map[name]}=={'particlegan/feature_cells.py'}
    assert all(sha(PERF/name)==digest for name,digest in ready['package_source_sha256'].items())
    assert len(ready['ast_splices'])==4
    splices=[]
    for splice in ready['ast_splices']:
        owner=splice.get('owner');node=find(composed,splice['name'],owner)
        assert ast_hash(node)==splice['proposal_ast_sha256']==ast_hash(find(perf,splice['name'],owner))
        if owner:
            original=find(count,splice['name'],owner)
            assert ast_hash(original)==splice['base_ast_sha256']
        else:
            assert splice['name']=='_group_integer_allocate' and splice['base_ast_sha256'] is None
        splices.append(dict(splice,decorators=[ast.unparse(d) for d in node.decorator_list]))
    # Revert declared methods inside the composed snapshot, preserving every
    # other statement/docstring/class attribute for a full class AST comparison.
    normalized_snapshot=deepcopy(find(composed,'FeatureCellSnapshot'))
    for name in ('_ordinary_mass_transport','_ordinary_support_transport','_ordinary_global_transport'):
        normalized_snapshot=replace_node(normalized_snapshot,find(normalized_snapshot,name),find(count,name,'FeatureCellSnapshot'))
    assert ast_hash(normalized_snapshot)==ast_hash(find(count,'FeatureCellSnapshot'))
    for name in ('ordinary_transport','cell_comparison','_mass_topology','select_parents'):
        assert ast_hash(find(composed,name,'FeatureCellSnapshot'))==ast_hash(find(count,name,'FeatureCellSnapshot'))
    # Independently reconstruct the intended backend settings and metadata.
    axis_setting=assignment(find(axis,'__init__','FeatureCellBirthDeath'),'settings')
    count_setting=assignment(find(count,'__init__','FeatureCellBirthDeath'),'settings')
    actual_setting=assignment(find(composed,'__init__','FeatureCellBirthDeath'),'settings')
    expected_setting=deepcopy(axis_setting)
    extra_settings={k.arg:k for k in count_setting.value.keywords if k.arg=='mass_policy' or k.arg.startswith('count_')}
    expected_by_name={k.arg:k for k in expected_setting.value.keywords}
    for name,keyword in extra_settings.items():
        if name in expected_by_name:
            expected_setting.value.keywords[expected_setting.value.keywords.index(expected_by_name[name])]=deepcopy(keyword)
        else:expected_setting.value.keywords.append(deepcopy(keyword))
    assert ast_hash(actual_setting)==ast_hash(expected_setting)
    expected_last=deepcopy(assignment(find(axis,'maybe_apply','FeatureCellBirthDeath'),'last'))
    count_last=assignment(find(count,'maybe_apply','FeatureCellBirthDeath'),'last')
    actual_last=assignment(find(composed,'maybe_apply','FeatureCellBirthDeath'),'last')
    original_keys={k.arg for k in expected_last.value.keywords}
    metadata={k.arg:k for k in count_last.value.keywords if k.arg.startswith('count_') or k.arg in
        {'ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves','ordinary_death_policy'}}
    assert set(metadata).isdisjoint(original_keys)
    expected_last.value.keywords.extend(deepcopy(list(metadata.values())))
    assert ast_hash(actual_last)==ast_hash(expected_last)
    # Undo only the intended composition additions. This reconstructs the
    # entire AXIS module, proving sampler/lineage/backend code is preserved.
    normalized=deepcopy(composed)
    normalized.body=[n for n in normalized.body if not (isinstance(n,ast.FunctionDef) and n.name=='_group_integer_allocate')]
    normalized=replace_node(normalized,find(normalized,'FeatureCellSnapshot'),find(axis,'FeatureCellSnapshot'))
    normalized=replace_node(normalized,assignment(find(normalized,'__init__','FeatureCellBirthDeath'),'settings'),axis_setting)
    normalized=replace_node(normalized,assignment(find(normalized,'maybe_apply','FeatureCellBirthDeath'),'last'),
                            assignment(find(axis,'maybe_apply','FeatureCellBirthDeath'),'last'))
    assert ast_hash(normalized)==ast_hash(axis)
    return dict(complete_axis_module_reconstructed=True,complete_count_snapshot_reconstructed=True,
                count_wrapper_comparison_topology_isolation_exact=True,all_other_axis_sources_exact=True,
                settings_ast_exact=True,diagnostic_metadata_ast_exact=True,ast_splices=splices,
                count_settings={name:ast.literal_eval(k.value) for name,k in extra_settings.items()},
                count_metadata_keys=list(metadata),training_sha256=candidate_map['particlegan/training.py'])


def runtime_checks(package,static):
    helper=load('ra4_composition_lineage_fixture',ROOT/'geometry/training-regression/lineage_checks.py')
    helper.setup(package,'cpu')
    import particlegan.training as training
    import particlegan.feature_cells as fc
    axis_fc=load('particlegan._composition_axis_fc',AXIS/'particlegan/feature_cells.py')
    axis_test=load('ra4_composition_axis_harness',HERE.parent/'test_axis_id.py')
    resolve,native=axis_test.frozen_harness_functions()
    options,_=resolve(SimpleNamespace(GANTrainer=training.GANTrainer),{}, {})
    assert options['evaluation_generate']=='indexed'
    assert inspect.signature(training.GANTrainer._generate).parameters['indices'].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    trainer=helper.make_trainer();bd=trainer.birth_death
    for name,value in static['count_settings'].items():assert bd.settings[name]==value
    assert bd.settings['latent_kernel']=='bounded_local_dv12_lineage'
    assert bd.settings['lineage_degree']==8 and bd.settings['latent_candidate_bound']==72
    bd._move(trainer,helper.ids([0,1]),helper.ids([300,301]))
    bd.lineage.validate(bd.lineage.neighbors)
    rows=helper.ids([0,300,1,301,500]);forwarded=[];original=bd.perturb_latent
    def observed(latent,*args,rows=None,**kw):
        forwarded.append(rows)
        return original(latent,*args,rows=rows,**kw)
    bd.perturb_latent=observed
    for ema in (False,True):
        model,prior=(trainer.ema_G,trainer.ema_prior) if ema else (trainer.G,trainer.prior)
        latent=prior.z[rows];a,b=helper.stream(),helper.stream()
        actual=native(trainer,model,latent,a,rows,options)
        assert forwarded[-1] is rows
        expected=trainer._generate(model,latent,0.,b,rows=rows)
        assert torch.equal(actual,expected) and torch.equal(a.get_state(),b.get_state())
        a,b=helper.stream(),helper.stream()
        actual=trainer.sample(17,ema=ema,generator=a)
        latent,indices=prior.sample(17,generator=b)
        expected=trainer._generate(model,latent,0.,b,indices)
        assert torch.equal(actual,expected) and torch.equal(a.get_state(),b.get_state())
        assert all(type(axis) is int for axis,_,_ in bd.latent_geometry._orders(prior.z))
    bd.perturb_latent=original
    saved=trainer.state_dict()
    assert saved['schema']==4 and saved['birth_death']['backend_schema']==4
    assert saved['birth_death']['settings']==bd.settings
    assert torch.equal(saved['birth_death']['lineage_neighbors'],bd.lineage.neighbors)
    buffer=io.BytesIO();torch.save(saved,buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cpu',weights_only=False)
    restored=helper.make_trainer();restored.load_state_dict(saved)
    assert helper.digest(restored.state_dict())==helper.digest(saved)
    assert not restored.birth_death.latent_geometry._entries and restored.birth_death.snapshot is None
    assert torch.equal(restored.birth_death.lineage.neighbors,bd.lineage.neighbors)
    before=helper.digest(trainer.state_dict())
    old_bd=axis_fc.FeatureCellBirthDeath(trainer,helper.SEED)
    bad_states=dict(old_ra3_mass_policy=old_bd.state_dict())
    bad=deepcopy(bd.state_dict());bad['settings']['count_family']='original_K_plus_support_2K_common_Q_over_3K'
    bad_states['old_3K_count_family']=bad
    bad=deepcopy(bd.state_dict());bad['settings']['mass_policy']='joint_mass_support_common_3K_unique_parents_v1'
    bad_states['old_3K_mass_policy']=bad
    for name,bad in bad_states.items():
        try:bd.load_state_dict(bad)
        except ValueError:pass
        else:raise AssertionError(name+' accepted')
        assert helper.digest(trainer.state_dict())==before,name+' mutated before rejection'
    # One original fixed identity/two-cluster reaction diagnostic, without a
    # gradient update or quality evaluation, exercises transplanted last fields.
    diagnostic=helper.make_trainer();diagnostic.birth_death.observe_real(helper.real_batch())
    last=diagnostic.birth_death.maybe_apply(diagnostic,0.)
    assert last is not None
    assert set(static['count_metadata_keys'])<=last.keys()
    assert last['count_multiplicity']==3*last['cells']+2
    assert last['count_cutoff']==.05/last['count_multiplicity']
    assert last['ordinary_mass_moves']+last['ordinary_support_moves']+last['ordinary_global_moves']==last['ordinary_moves']
    assert diagnostic.completed_steps==0
    return dict(canonical_harness_indexed=True,fifth_positional_ids_forwarded=True,
                live_ema_sample_bits_and_rng_exact=True,cached_axes_python_ints=True,
                graph_schema=4,trainer_schema=4,composed_settings_serialized=True,
                exact_graph_checkpoint_roundtrip=True,derived_caches_discarded=True,
                incompatible_count_states_rejected_atomically=list(bad_states),
                fixed_reaction_metadata_exercised=True,gradient_updates=0,new_seeds=0,
                fixed_reaction=dict(cells=last['cells'],multiplicity=last['count_multiplicity'],cutoff=last['count_cutoff'],
                    mass_moves=last['ordinary_mass_moves'],support_moves=last['ordinary_support_moves'],
                    global_moves=last['ordinary_global_moves']),
                fixture_source_sha256=sha(ROOT/'geometry/training-regression/lineage_checks.py'),
                fixture_input_sha256=sha(ROOT/'geometry/gpu-inputs.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,default=ROOT/'pkg-CB64-RA4')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Evidence already exists')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    before={str(p):sources(p) for p in (args.package_root,AXIS,COUNT,PERF)}
    proof_sha,compose_sha,ready_sha=sha(PROOF),sha(COMPOSE),sha(PERF_READY)
    start=time.perf_counter()
    static=static_checks(args.package_root);runtime=runtime_checks(args.package_root,static)
    assert before=={str(p):sources(p) for p in (args.package_root,AXIS,COUNT,PERF)}
    assert (proof_sha,compose_sha,ready_sha)==(sha(PROOF),sha(COMPOSE),sha(PERF_READY))
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS',scope='independent read-only composed count/sampler/API/checkpoint CPU audit; no quality verdict',
        cpu_threads=1,cuda_initialized=False,seconds=time.perf_counter()-start,static=static,runtime=runtime,
        source_sha256=before,composition_sha256=proof_sha,compose_script_sha256=compose_sha,
        planner_ready_sha256=ready_sha,script_sha256=sha(__file__),harness_sha256=sha(HARNESS))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',output=str(args.output),seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
