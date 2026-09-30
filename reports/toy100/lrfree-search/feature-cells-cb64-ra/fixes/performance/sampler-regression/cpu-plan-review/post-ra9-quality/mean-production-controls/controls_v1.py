"""Focused CPU RA10 ownership controls on the owner's sealed packet archive.

The packet archive is mechanical evidence, not a resumable trainer checkpoint.
Its exact contents are checked before binding a new-process pointer/version epoch.
No complete planner or quality emission runs here.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path, *, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'retain earlier control attempts'
    prep = json.loads(args.preparation.read_text())
    assert prep['status'] == 'FROZEN_CPU_OWNERSHIP_CONTROL_INPUTS'
    for path, digest in prep['source_and_input_sha256'].items():
        assert sha(path) == digest, path
    assert prep['source_and_input_sha256'][str(Path(__file__).resolve())] == sha(Path(__file__))
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    sys.dont_write_bytecode = True
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    package = Path(prep['package_root'])
    loaded = load_module('ra10_ownership_package', package/'particlegan/__init__.py', package=True)
    fc = sys.modules['ra10_ownership_package.feature_cells']
    mt = sys.modules['ra10_ownership_package.mean_transport']
    skeleton = load_module('ra10_ownership_skeleton', Path(__file__).with_name('controls_skeleton.py'))
    archive = torch.load(prep['packet_archive'], map_location='cpu', weights_only=False)
    before = torch.load(prep['before_checkpoint'], map_location='cpu', weights_only=False)
    after = torch.load(prep['after_checkpoint'], map_location='cpu', weights_only=False)
    hook = torch.load(prep['hook_state'], map_location='cpu', weights_only=False)
    trace = json.loads(Path(prep['trace']).read_text())
    assert archive['schema'] == 1 and 'not checkpoint' in archive['purpose']
    assert before['device'] == after['device'] == 'cpu'
    assert before['birth_death']['backend_schema'] == after['birth_death']['backend_schema'] == 9
    assert before['completed_steps'] == 0 and after['completed_steps'] == 1
    global_rng = torch.get_rng_state().clone()
    results = []

    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key:clone(item) for key,item in value.items()}
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        return deepcopy(value)

    def literal_linear(weight, bias):
        module = nn.Linear.__new__(nn.Linear)
        nn.Module.__init__(module)
        module.in_features, module.out_features = weight.shape[1], weight.shape[0]
        module.weight, module.bias = nn.Parameter(weight.clone()), nn.Parameter(bias.clone())
        return module

    def network(weights):
        return nn.Sequential(literal_linear(weights['0.weight'], weights['0.bias']), nn.LeakyReLU(.2),
            literal_linear(weights['2.weight'], weights['2.bias']), nn.LeakyReLU(.2),
            literal_linear(weights['4.weight'], weights['4.bias']))

    namespace = {'torch':torch,'nn':nn}
    host = Path(prep['native_host_source'])
    nodes = [node for node in ast.parse(host.read_text()).body
             if isinstance(node, ast.ClassDef) and node.name == 'SimpleMLPDiscriminator']
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(host),'exec'),namespace)
    native_D = namespace['SimpleMLPDiscriminator']

    def models(weights):
        roots = {}
        for name in ('G','ema_G'):
            value = weights[name]
            roots[name] = literal_linear(value['weight'],value['bias']) if 'weight' in value else network(value)
        value = weights['D']
        if 'freqs' not in value:
            roots['D'] = network(value)
        else:
            module = native_D.__new__(native_D)
            nn.Module.__init__(module)
            module.fourier = len(value['freqs'])
            module.register_buffer('freqs',value['freqs'].clone())
            ids = sorted(int(key.split('.')[1]) for key in value if key.endswith('.weight'))
            layers = []
            for index in ids:
                layers.append(literal_linear(value[f'net.{index}.weight'],value[f'net.{index}.bias']))
                if index != ids[-1]: layers.append(nn.LeakyReLU(.2,inplace=True))
            module.net = nn.Sequential(*layers)
            roots['D'] = module
        return roots

    def generator(state):
        result = torch.Generator(device='cpu')
        result.set_state(state.clone())
        return result

    def context():
        roots = models(archive['models'])
        prior = loaded.ParticlePrior.__new__(loaded.ParticlePrior)
        nn.Module.__init__(prior)
        prior.z = nn.Parameter(archive['state']['fast'].clone())
        ema = loaded.ParticlePrior.__new__(loaded.ParticlePrior)
        nn.Module.__init__(ema)
        ema.z = nn.Parameter(archive['state']['ema'].clone())
        graph = archive['state']['lineage']
        lineage = fc.LatentLineage(len(graph),graph.shape[1],torch.device('cpu'))
        lineage.neighbors.copy_(graph)
        lineage.validate(lineage.neighbors)
        backend = fc.FeatureCellBirthDeath.__new__(fc.FeatureCellBirthDeath)
        backend.lineage = lineage
        backend.stream = generator(archive['state']['stream'])
        backend.sample_shape = tuple(before['birth_death']['sample_shape'])
        backend.settings = deepcopy(before['birth_death']['settings'])
        backend._linears = [module for module in roots['D'].modules() if isinstance(module,nn.Linear)]
        backend._heads = None
        trainer = SimpleNamespace(**roots,prior=prior,ema_prior=ema,device=torch.device('cpu'),
            _STREAMS=tuple(before['streams']),controller=SimpleNamespace(latent_bandwidth=archive['state']['bandwidth'].clone()),
            opt_g=SimpleNamespace(state={prior.z:clone(archive['state']['row_state'])},
                                 latent_history=clone(archive['state']['history'])))
        assert len(trainer._STREAMS) == 4
        for name,state in before['streams'].items(): setattr(trainer,name,generator(state))
        state = mt.copy_state(backend,trainer)
        state.owned_streams = [getattr(trainer,name) for name in trainer._STREAMS]
        snapshot = fc.FeatureCellSnapshot.__new__(fc.FeatureCellSnapshot)
        snapshot.__dict__.update(clone(archive['snapshot']))
        assert snapshot.device == torch.device('cpu')
        values = clone(archive['packet'])
        packet = mt.CopyPacket(expected_epoch=mt.epoch(state,snapshot),
            fingerprint=archive['packet_fingerprint'],**values)
        assert packet.content_digest() == archive['packet_fingerprint']
        return backend,trainer,state,snapshot,packet

    def protected(state,snapshot):
        return mt.tensor_digest(state.prior.z,state.ema_prior.z,state.lineage.neighbors,
            state.row_state,state.history,state.bandwidth,state.stream.get_state(),
            snapshot.row_features,snapshot.query_cell_ids,tuple(sorted(state.consumed)),
            *state.model_tensors,state.buffer_epoch(),snapshot.cache_version,
            *(stream.get_state() for stream in state.owned_streams))

    def reject(name, alter_packet=None, alter_state=None):
        backend,trainer,state,snapshot,packet = context()
        if alter_state is not None:
            with torch.no_grad(): alter_state(state,snapshot,trainer,packet)
        if alter_packet is not None:
            packet = alter_packet(packet,state,snapshot)
            packet = replace(packet,fingerprint=packet.content_digest())
        old = protected(state,snapshot)
        try:
            mt.commit_packet(packet,state,snapshot)
        except (ValueError,RuntimeError):
            pass
        else:
            raise AssertionError('invalid packet was accepted: '+name)
        assert protected(state,snapshot) == old,name
        results.append({'control':name,'status':'PASS','reject_before_write':True})

    # One fixed preview replay, never another pair planner or chart fit.
    parsed = ast.parse((package/'particlegan/mean_transport.py').read_text())
    preview_node = next(node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name=='preview_pairs')
    draws = [node for node in ast.walk(preview_node) if isinstance(node,ast.Call)
             and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
             and node.func.value.id=='torch' and node.func.attr=='randn']
    assert len(draws) == 1
    assert not any(draw in list(ast.walk(loop)) for loop in ast.walk(preview_node)
                   if isinstance(loop,(ast.For,ast.While)) for draw in draws)
    results.append({'control':'shared_single_preview_draw','status':'PASS',
                    'evidence':'frozen production source and exact fixed-preview replay below'})

    bd,tr,state,snapshot,original_packet = context()
    pc=archive['preview_context']
    fixed=mt.FixedMoment(**clone(pc['fixed']))
    fast_view=mt.View(None,**clone(pc['fast']))
    ema_view=mt.View(None,**clone(pc['ema']))
    pairs=mt.CandidatePairs(**clone(pc['pairs']))
    state.stream.set_state(pc['pre_draw_stream'])
    geometry=fc.BoundedLatentGeometry(rank=bd.settings['latent_rank'],
        neighbors=bd.settings['latent_neighbors'],chunk=bd.settings['chunk'],lineage=bd.lineage)
    draw_calls=[]
    original_randn=torch.randn
    def traced_randn(*values,**options):
        if options.get('generator') is state.stream:
            draw_calls.append(tuple(values[0]))
        return original_randn(*values,**options)
    torch.randn=traced_randn
    try:
        replay,detail=mt.preview_pairs(snapshot,fixed,fast_view,ema_view,pairs,state,stream=state.stream,
            features_fast=lambda x:mt.capture_features(bd,tr,tr.G,x),
            features_ema=lambda x:mt.capture_features(bd,tr,tr.ema_G,x),geometry=geometry)
    finally:
        torch.randn=original_randn
    assert replay is not None
    assert draw_calls==[(len(pairs.children),state.prior.z.shape[1])]
    assert replay.fingerprint==original_packet.fingerprint
    assert replay.content_digest()==archive['packet_fingerprint']
    assert torch.equal(state.stream.get_state(),archive['state']['stream'])
    assert detail['paired_noise_draws']==1 and detail['no_redraw']
    assert detail['attempts']==trace['preview_detail']['attempts']
    assert detail['accepted']==trace['preview_detail']['accepted']
    assert detail['objective_after_virtual']==trace['preview_detail']['objective_after_virtual']
    results.append({'control':'fixed_preview_exact_coordinates_features_inheritance_future_RNG',
        'status':'PASS','preview_rows':len(pairs.children),'accepted_rows':len(replay.children),
        'shared_noise_draws':len(draw_calls),'planner_replay':False})

    # Independently apply actual chart predicates to the exact prepared features.
    children,parents=original_packet.children,original_packet.parents
    actual_fast=mt.observe_view(snapshot,original_packet.fast_features,coordinates=original_packet.fast_coordinates)
    actual_ema=mt.observe_view(snapshot,original_packet.ema_features,coordinates=original_packet.ema_coordinates)
    for old,new in ((fast_view,actual_fast),(ema_view,actual_ema)):
        assert bool(old.eligible[children].all() & old.eligible[parents].all() & new.eligible.all())
        assert bool((old.pvalues[children]>.05).all() & (old.pvalues[parents]>.05).all() & (new.pvalues>.05).all())
        assert bool((old.categories[children].remainder(2)==0).all()
            &(old.categories[parents].remainder(2)==0).all()&(new.categories.remainder(2)==0).all())
        assert torch.equal(old.cells[children],old.cells[parents])
        assert torch.equal(new.cells,old.cells[children])
        assert torch.equal(new.categories,old.categories[children])
        assert torch.equal(new.groups,old.groups[children])
    assert torch.equal(actual_fast.groups,actual_ema.groups)
    assert len(torch.unique(children))==len(children) and len(torch.unique(parents))==len(parents)
    assert not bool(torch.isin(children,parents).any())
    assert not bool(torch.isin(torch.cat((children,parents)),pc['reserved_rows']).any())
    assert not bool(torch.isin(children,pairs.protected_rows).any())
    assert pairs.budget==math.floor(.05*len(state.prior.z))-pairs.earlier_ordinary
    assert len(pairs.children)<=pairs.budget and len(children)<=len(pairs.children)
    pair_map=dict(zip(pairs.children.tolist(),pairs.parents.tolist()))
    assert all(pair_map[c]==p for c,p in zip(children.tolist(),parents.tolist()))
    results.append({'control':'actual_both_view_support_inside_cell_category_group_reservations_budget',
                    'status':'PASS','accepted_rows':len(children)})

    backend,trainer,state,snapshot,packet = context()
    old_epoch = mt.epoch(state,snapshot)
    with torch.no_grad():
        for root in (trainer.G,trainer.ema_G,trainer.D):
            for buffer in root.buffers(): buffer.copy_(buffer)
    assert mt.epoch(state,snapshot) == old_epoch,'neutral buffer copy version falsely changed epoch'
    fast,ema = state.prior.z.detach().clone(),state.ema_prior.z.detach().clone()
    old_rows,old_history = clone(state.row_state),clone(state.history)
    old_stream = state.stream.get_state().clone()
    called = []
    original_graph,original_refresh = state.lineage.register_copies,snapshot.refresh_rows
    def register(children,parents):
        called.append('lineage')
        return original_graph(children,parents)
    def refresh(children,features):
        called.append('refresh')
        return original_refresh(children,features)
    state.lineage.register_copies,snapshot.refresh_rows = register,refresh
    applied = mt.commit_packet(packet,state,snapshot)
    assert called == ['lineage','refresh']
    assert torch.equal(state.stream.get_state(),old_stream)
    assert torch.equal(state.prior.z[packet.children],packet.fast_coordinates)
    assert torch.equal(state.ema_prior.z[packet.children],packet.ema_coordinates)
    untouched = torch.ones(len(fast),dtype=torch.bool)
    untouched[packet.children] = False
    assert torch.equal(state.prior.z[untouched],fast[untouched])
    assert torch.equal(state.ema_prior.z[untouched],ema[untouched])
    for key,value in old_rows.items():
        if isinstance(value,torch.Tensor) and value.shape == fast.shape:
            assert torch.equal(state.row_state[key][untouched],value[untouched])
            assert torch.equal(state.row_state[key][packet.children],packet.parent_row_state[key])
        else:
            assert mt.tensor_digest(state.row_state[key]) == mt.tensor_digest(value)
    if old_history is not None:
        assert torch.equal(state.history[untouched],old_history[untouched])
        assert torch.equal(state.history[packet.children],packet.parent_history)
    state.lineage.validate(state.lineage.neighbors)
    assert packet.fingerprint in state.consumed
    assert sorted(applied['moved_rows']) == packet.children.sort().values.tolist()
    results.append({'control':'exact_commit_untouched_inheritance_cache_graph_no_draw',
                    'status':'PASS','accepted_rows':len(packet.children),'new_process_epoch_explicit':True})

    reject('stale_FAST',alter_state=lambda s,c,t,p:s.prior.z.add_(0))
    reject('stale_chart',alter_state=lambda s,c,t,p:c.centers.add_(0))
    reject('stale_lineage',alter_state=lambda s,c,t,p:s.lineage.neighbors.copy_(s.lineage.neighbors))
    reject('stale_model',alter_state=lambda s,c,t,p:next(t.G.parameters()).add_(0))
    reject('stale_bandwidth',alter_state=lambda s,c,t,p:s.bandwidth.add_(0))
    reject('stale_post_draw_stream',alter_state=lambda s,c,t,p:torch.rand((),generator=s.stream))
    reject('consumed',alter_state=lambda s,c,t,p:s.consumed.add(p.fingerprint))
    reject('malformed_coordinate_shape',alter_packet=lambda p,s,c:replace(p,fast_coordinates=p.fast_coordinates[:,:-1]))
    reject('malformed_row_dtype',alter_packet=lambda p,s,c:replace(p,children=p.children.double()))
    def out_of_bounds(p,s,c):
        children=p.children.clone(); children[0]=len(s.prior.z)
        return replace(p,children=children)
    reject('malformed_row_bounds',alter_packet=out_of_bounds)
    def overlapping(p,s,c):
        parents=p.parents.clone(); parents[0]=p.children[0]
        return replace(p,parents=parents)
    reject('overlapping_source_child',alter_packet=overlapping)
    def duplicate_child(p,s,c):
        children=p.children.clone(); children[1]=children[0]
        return replace(p,children=children)
    reject('duplicate_children',alter_packet=duplicate_child)
    def nonfinite_coordinates(p,s,c):
        values=p.ema_coordinates.clone(); values[0,0]=float('nan')
        return replace(p,ema_coordinates=values)
    reject('nonfinite_coordinates',alter_packet=nonfinite_coordinates)
    reject('malformed_feature_shape',alter_packet=lambda p,s,c:replace(p,fast_features=p.fast_features[:,:-1]))
    def missing_row(p,s,c):
        rows=clone(p.parent_row_state); assert rows
        rows.pop(next(iter(rows)))
        return replace(p,parent_row_state=rows)
    reject('malformed_optimizer_inheritance',alter_packet=missing_row)
    if packet.parent_history is not None:
        reject('malformed_history',alter_packet=lambda p,s,c:replace(p,parent_history=p.parent_history[:-1]))

    # Actual wrappers and registered-state neutrality on literal dirty modules.
    def observation_case(*, raises=False, guarded=True):
        bd,tr,_,_,_ = context()
        streams=[bd.stream,*[getattr(tr,name) for name in tr._STREAMS]]
        roots=[]
        for name in ('G','ema_G','D'):
            original=getattr(tr,name)
            dirty=skeleton.dirty_eval_module(torch,streams,raise_after=raises and name=='G')
            dirty.inner=original
            original_forward=dirty.forward
            def wrapped(inputs, forward=original_forward, inner=original):
                return inner(forward(inputs))
            dirty.forward=wrapped
            setattr(tr,name,dirty); roots.append(dirty)
        bd._linears=[module for module in tr.D.modules() if isinstance(module,nn.Linear)]
        bd._heads=None
        old=skeleton.owned_snapshot(torch,roots,streams)
        with torch.random.fork_rng(devices=[]):
            if guarded:
                try:
                    output=mt.capture_features(bd,tr,tr.G,tr.prior.z[:2])
                    assert output is not None
                except RuntimeError as error:
                    assert raises and str(error)=='intentional dirty eval exception'
                skeleton.assert_owned_restored(torch,old)
            else:
                tr.G(tr.prior.z[:2])
                assert not torch.equal(torch.get_rng_state(),old['global_cpu'])
                assert not torch.equal(streams[0].get_state(),old['streams'][0][1])
                assert 'extra' in tr.G._buffers and tr.G.first.grad is not old['gradients'][0][1]
        results.append({'control':'dirty_observation_'+('exception' if raises else 'normal') if guarded else 'dirty_unguarded_control',
                        'status':'PASS','registered_objects_values_modes_grads_global_owned_rng':guarded})
    observation_case(guarded=False)
    observation_case()
    observation_case(raises=True)

    # Bind the actual owner reaction/hook, comparing full original phase rows.
    def hook_checks(fresh_before,fresh_after,sidecar,hook_state,label):
        event=fresh_after['birth_death']['last']
        old_children=[row for call in sidecar['old_copy_calls'] for row in call['children']]
        old_parents=[row for call in sidecar['old_copy_calls'] for row in call['parents']]
        moved=old_children+sidecar['novel_children']+sidecar['mean_children']
        sources=old_parents+sidecar['novel_source_seeds']+sidecar['mean_parents']
        assert len(set(moved))==len(moved) and len(set(sources))==len(sources)
        assert not set(moved)&set(sources)
        assert sorted(moved)==sorted(sidecar['moved_rows'])
        assert sorted(set(old_children+old_parents+sidecar['novel_children']+sidecar['novel_source_seeds']))==sidecar['prefix_reserved']
        assert not (set(sidecar['mean_children'])|set(sidecar['mean_parents']))&set(sidecar['prefix_reserved'])
        assert event['ordinary_copy_moves']==sum(event[k] for k in ('ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves','ordinary_mean_moves'))
        assert event['ordinary_moves']==event['ordinary_copy_moves']+event['ordinary_novel_birth_moves']<=math.floor(.05*len(fresh_before['models']['prior']['z']))
        assert event['moves']==event['ordinary_moves']+event['iso_moves']==len(moved)
        assert len(sidecar['mean_children'])==event['ordinary_mean_moves']
        assert event['ordinary_action_kinds'].count(4)==len(sidecar['mean_children'])
        assert fresh_before['completed_steps']==0 and fresh_after['completed_steps']==1
        assert event['paired_average']['step']==1 and event['paired_average']['snapshot']==fresh_after['birth_death']['snapshot_serial']
        keep=torch.ones(len(fresh_before['models']['prior']['z']),dtype=torch.bool)
        keep[torch.tensor(moved,dtype=torch.long)]=False
        for name in ('prior','ema_prior'):
            assert torch.equal(fresh_after['models'][name]['z'][keep],fresh_before['models'][name]['z'][keep])
        for name in ('G','D','ema_G'):
            assert mt.tensor_digest(fresh_after['models'][name])==mt.tensor_digest(fresh_before['models'][name])
        assert all(torch.equal(fresh_after['streams'][name],value) for name,value in fresh_before['streams'].items())
        assert sidecar['caller_hooks']
        assert all(sorted(call['rows'])==sorted(moved) for call in sidecar['caller_hooks'])
        idx=torch.tensor(moved,dtype=torch.long)
        evidence_old,evidence_new=hook_state['evidence_before'],hook_state['evidence_after']
        if evidence_new is not None:
            assert evidence_new['counters']['resets']==evidence_old['counters']['resets']+len(moved)
            for name in ('M','Qs','W','S','flag'):
                assert not bool(evidence_new[name][idx].any())
                assert torch.equal(evidence_new[name],fresh_after['row_evidence'][name])
        tester=hook_state['tester_after']
        if tester is not None:
            assert 'rebase' not in tester
            assert not bool(tester['stationary_rows'][idx].any())
            assert bool(tester['invalid_block_rows'][idx].all())
            assert torch.equal(tester['anchor'].view(len(keep),-1)[idx],fresh_after['models']['prior']['z'][idx])
        return {'case':label,'status':'PASS','moved_rows':moved,'source_rows':sources,
            'phase_counts':{k:event[k] for k in ('ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves',
                'ordinary_mean_moves','ordinary_novel_birth_moves','ordinary_copy_moves','ordinary_moves','iso_moves','moves')},
            'caller_hooks':sidecar['caller_hooks'],'fresh_zero_initial_evidence':True,
            'complete_own_evidence_and_participation_reset':True,'untouched_source_model_stream_bytes':True}
    hooked=[hook_checks(before,after,trace,hook,'grid')]
    toy_before=torch.load(prep['toy_before_checkpoint'],map_location='cpu',weights_only=False)
    toy_after=torch.load(prep['toy_after_checkpoint'],map_location='cpu',weights_only=False)
    toy_hook=torch.load(prep['toy_hook_state'],map_location='cpu',weights_only=False)
    toy_trace=json.loads(Path(prep['toy_trace']).read_text())
    assert toy_after['birth_death']['last']['mean_transport']['status']=='veto'
    assert toy_after['birth_death']['last']['mean_transport']['attempts']==0
    hooked.append(hook_checks(toy_before,toy_after,toy_trace,toy_hook,'toy'))
    results.append({'control':'genuine_fresh_reaction_complete_phase_hook_reset_rebase_final_lease',
                    'status':'PASS','cases':hooked})

    # All source and fixture bytes must still match; no source mutation or CUDA.
    assert torch.equal(torch.get_rng_state(),global_rng)
    assert not torch.cuda.is_initialized()
    for path,digest in prep['source_and_input_sha256'].items(): assert sha(path)==digest,path
    result={'status':'PASS','controls':results,'source_and_input_sha256':prep['source_and_input_sha256'],
        'preparation_sha256':sha(args.preparation),'cuda_initialized':False,
        'fresh_backend9_fixture':True,'new_training_steps':0,'new_quality_emissions':0,
        'planner_replays':0,'scope':'CPU packet/observer ownership mechanics; no GPU or quality qualification',
        'phase_hook_reservation_checks_pending_final_trace_binding':False,
        'hook_evidence':hooked,'fixed_preview_replays':1,'no_synthetic_positive_stamp':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':'PASS','controls':len(results),'receipt_sha256':sha(args.output)}))


if __name__=='__main__':
    main()
