"""Corrected raw-input fixture v2, fresh backend9 reaction mechanics; no quality score."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
API = ROOT / 'quality/ra8/integration-contract/check_api.py'
HOST = Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100/toy_models.py')
CASES = (('grid', ROOT / 'validation-cb64-ra9/screens/runs/grid100/final-state.pt'),
         ('toy', ROOT / 'validation-cb64-ra9/learned/training/toy/CB64-RA9/checkpoint-2000.pt'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    seal = json.loads((HERE / 'SOURCE-FROZEN-v2.json').read_text())
    for path, digest in seal['source_and_input_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'protected byte mismatch: {path}')
    return seal


def extract(path, names, namespace):
    nodes = [n for n in ast.parse(Path(path).read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('frozen fixture symbol missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def log(event, **values):
    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), event=event, **values)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--package-root', type=Path, default=HERE / 'pkg-MEAN')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('retain previous attempts; output exists')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    sys.dont_write_bytecode = True
    seal = verify()  # Raw guards precede Torch/PT interpretation and constructors.
    if args.package_root.resolve() != (HERE / 'pkg-MEAN').resolve():
        raise ValueError('mechanics package must be the exact owner frozen package')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.2, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('mean_owner', args.package_root / 'particlegan/__init__.py',
        submodule_search_locations=[str(args.package_root / 'particlegan')])
    package = importlib.util.module_from_spec(spec)
    sys.modules['mean_owner'] = package
    spec.loader.exec_module(package)
    fc = sys.modules['mean_owner.feature_cells']
    mt = sys.modules['mean_owner.mean_transport']
    namespace = dict(torch=torch, nn=nn, deepcopy=deepcopy)
    extract(API, ('linear', 'network'), namespace)
    extract(HOST, ('SimpleMLPDiscriminator',), namespace)
    linear, network = namespace['linear'], namespace['network']
    host_D = namespace['SimpleMLPDiscriminator']

    def cpu(value):
        if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
        if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list): return [cpu(v) for v in value]
        if isinstance(value, tuple): return tuple(cpu(v) for v in value)
        return deepcopy(value)

    def to_device(value):
        if isinstance(value, torch.Tensor): return value.detach().clone().to(device)
        if isinstance(value, dict): return {k: to_device(v) for k, v in value.items()}
        if isinstance(value, list): return [to_device(v) for v in value]
        if isinstance(value, tuple): return tuple(to_device(v) for v in value)
        return deepcopy(value)

    def state_hash(value):
        h = hashlib.sha256()
        def visit(v):
            if isinstance(v, torch.Tensor):
                h.update(f'{v.dtype}:{tuple(v.shape)}'.encode())
                h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(v, dict):
                h.update(b'dict')
                for key in sorted(v, key=repr): visit(key); visit(v[key])
            elif isinstance(v, (list, tuple)):
                h.update(type(v).__name__.encode())
                for child in v: visit(child)
            else: h.update(f'{type(v).__name__}:{v!r}'.encode())
        visit(value)
        return h.hexdigest()

    def construct(saved, label):
        weights = to_device(saved['models'])
        if label == 'toy':
            G, D = network(weights['G']), network(weights['D'])
        else:
            G = linear(weights['G']['weight'], weights['G']['bias'])
            D = host_D.__new__(host_D)
            nn.Module.__init__(D)
            D.fourier = len(weights['D']['freqs'])
            D.register_buffer('freqs', weights['D']['freqs'].clone())
            ids = sorted(int(k.split('.')[1]) for k in weights['D'] if k.endswith('.weight'))
            layers = []
            for index in ids:
                layers.append(linear(weights['D'][f'net.{index}.weight'], weights['D'][f'net.{index}.bias']))
                if index != ids[-1]: layers.append(nn.LeakyReLU(.2, inplace=True))
            D.net = nn.Sequential(*layers)
        prior = package.ParticlePrior.__new__(package.ParticlePrior)
        nn.Module.__init__(prior)
        prior.z = nn.Parameter(weights['prior']['z'].clone())
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            trainer = package.GANTrainer(package.Recipe(**deepcopy(saved['recipe'])), G, D, prior=prior,
                seed=314159, optimizer_options=deepcopy(saved['optimizer_options']),
                penalty_options=deepcopy(saved['penalty_options']), serial_backward=saved.get('serial_backward', False))
        # New-law construction. Only explicitly named raw tensor inputs are copied;
        # no old trainer/backend/settler/controller checkpoint is loaded or relabeled.
        # The native optimizer factory initializes supplied G/D. Restore the
        # explicit frozen raw inputs AFTER construction, preserving backend9.
        trainer.G.load_state_dict(weights['G'])
        trainer.D.load_state_dict(weights['D'])
        trainer.ema_D.load_state_dict(weights['D'])  # Fresh critic anchor matches copied current D.
        trainer.ema_G.load_state_dict(weights['ema_G'])
        trainer.ema_prior.load_state_dict(weights['ema_prior'])
        trainer.controller.latent_bandwidth.copy_(saved['controller']['latent_bandwidth'].to(device))
        if trainer.log_output_sigma is not None:
            trainer.log_output_sigma.copy_(saved['output_noise']['log_sigma'].to(device))
        rows = [v for v in saved['optimizers'][0]['state'].values()
                if any(isinstance(x, torch.Tensor) and x.shape == prior.z.shape for x in v.values())]
        if len(rows) != 1: raise AssertionError('exact one frozen prior optimizer row-state source required')
        trainer.opt_g.state[trainer.prior.z] = to_device(rows[0])
        latent = saved['optimizers'][0]['regularizer']['latent']
        if trainer.opt_g.latent_history is not None:
            if latent is None: raise AssertionError('missing frozen copy history')
            trainer.opt_g.latent_history.copy_(latent['history'].to(device))
        trainer.last_output_sigma = (float(saved['output_noise']['log_sigma'].exp())
                                    if trainer.log_output_sigma is not None else float(trainer.recipe.output_noise_std))
        bd = trainer.birth_death
        if bd.BACKEND_SCHEMA != 9 or trainer.completed_steps != 0:
            raise AssertionError('not a fresh backend9 construction')
        # Exact original FIFO is fed through the original public ingestion law.
        bd.observe_real(saved['birth_death']['reservoir'].to(device))
        for name in ('G','D','prior','ema_G','ema_prior'):
            if state_hash(getattr(trainer,name).state_dict())!=state_hash(saved['models'][name]):
                raise AssertionError(f'raw source model/table binding failed: {name}')
        if (state_hash(bd.reservoir)!=state_hash(saved['birth_death']['reservoir'])
                or state_hash(trainer.controller.latent_bandwidth)!=state_hash(saved['controller']['latent_bandwidth'])
                or state_hash(trainer.opt_g.state[trainer.prior.z])!=state_hash(rows[0])
                or (trainer.opt_g.latent_history is not None and state_hash(trainer.opt_g.latent_history)!=state_hash(latent['history']))
                or (trainer.log_output_sigma is not None and state_hash(trainer.log_output_sigma)!=state_hash(saved['output_noise']['log_sigma']))):
            raise AssertionError('named raw FIFO/bandwidth/moment/history/noise binding failed')
        if trainer.lr_settle is not None:
            for group,tester in trainer.lr_settle.pairs((trainer.opt_g,trainer.opt_d)):
                tester.begin(group['params'])  # Original pre-update anchor boundary, no observation/clock advance.
        return trainer

    # Execute the literal unchanged caller hook/body, without a GAN gradient step.
    tree = ast.parse((args.package_root / 'particlegan/training.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GANTrainer')
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_step')
    hook = next(n for n in step.body if isinstance(n, ast.If)
                and isinstance(n.test, ast.Compare) and 'self.birth_death is not None' == ast.unparse(n.test)
                and any(isinstance(child,ast.Assign) and any(ast.unparse(t)=='event' for t in child.targets)
                        for child in n.body))
    event_hook = next(n for n in hook.body if isinstance(n, ast.If))
    complete = next(n for n in step.body if isinstance(n, ast.AugAssign)
                    and ast.unparse(n.target) == 'self.completed_steps')
    serve = next(n for n in step.body if isinstance(n, ast.Expr)
                 and ast.unparse(n.value) == 'self._serve_apply()')
    hook_code = compile(ast.Module(body=[event_hook, complete, serve], type_ignores=[]), 'unchanged_trainer_hook', 'exec')

    global_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    records = []
    try:
        with torch.no_grad():
            for label, path in CASES:
                started = time.perf_counter()
                log('fresh_reaction_start', case=label, device=str(device))
                source = torch.load(path, map_location='cpu', weights_only=False)['trainer']
                original_hash = state_hash(source)
                trainer = construct(source, label)
                bd = trainer.birth_death
                case_dir = args.output.parent / label
                if case_dir.exists(): raise ValueError('retain prior case directory')
                case_dir.mkdir()
                before = trainer._state_dict()
                bd.check_state(before['birth_death'])
                torch.save(before, case_dir / 'before.pt')
                streams_before = {name: getattr(trainer, name).get_state().clone() for name in trainer._STREAMS}
                model_before = {name: state_hash(getattr(trainer, name).state_dict()) for name in ('G','D','ema_G')}
                captured, old_copies, hook_calls = {}, [], []
                original_transport = fc.FeatureCellSnapshot.ordinary_transport
                original_mean = fc.run_mean_phase
                original_commit = mt.commit_packet
                original_preview = mt.preview_pairs
                original_move = bd._move
                move_owned = ('_move' in bd.__dict__, bd.__dict__.get('_move'))

                def transport(snapshot, q, flags, comparison, **kwargs):
                    result = original_transport(snapshot, q, flags, comparison, **kwargs)
                    captured['ordinary'] = cpu(result[2])
                    captured['ordinary_copy_children'] = cpu(result[0])
                    captured['ordinary_copy_parents'] = cpu(result[1])
                    return result

                def preview_call(snapshot,fixed,fast,ema,pairs,context,**kwargs):
                    captured['preview_context']=dict(fixed={k:cpu(v) for k,v in vars(fixed).items()},
                        fast={k:cpu(v) for k,v in vars(fast).items() if k!='features'},
                        ema={k:cpu(v) for k,v in vars(ema).items() if k!='features'},
                        pairs={k:cpu(v) for k,v in vars(pairs).items()},
                        pre_draw_stream=cpu(context.stream.get_state()),reserved_rows=captured['reserved_rows'])
                    result=original_preview(snapshot,fixed,fast,ema,pairs,context,**kwargs)
                    captured['preview_detail']=cpu(result[1])
                    return result

                def copy_call(owner, children, parents):
                    old_copies.append((cpu(children), cpu(parents)))
                    return original_move(owner, children, parents)

                def mean_call(backend, owner, snapshot, fixed, stamp, **kwargs):
                    captured['reserved_rows'] = cpu(torch.unique(kwargs['reserved_rows']))
                    captured['earlier_ordinary'] = kwargs['earlier_ordinary']
                    result = original_mean(backend, owner, snapshot, fixed, stamp, **kwargs)
                    captured['mean_children'], captured['mean_parents'] = cpu(result['children']), cpu(result['parents'])
                    return result

                def commit_call(packet, context, snapshot):
                    # Mechanical ownership artifact, never a resumable trainer state.
                    # Pointer/version epoch is rebound only by the independent control;
                    # stored coordinate and row-state bytes are not regenerated.
                    archive = dict(schema=1, purpose='focused new-process packet controls; not checkpoint/replay',
                        snapshot={k: cpu(v) for k,v in snapshot.__dict__.items()},
                        state=dict(fast=cpu(context.prior.z), ema=cpu(context.ema_prior.z),
                            lineage=cpu(context.lineage.neighbors), row_state=cpu(context.row_state),
                            history=cpu(context.history), bandwidth=cpu(context.bandwidth), stream=cpu(context.stream.get_state())),
                        packet={k:cpu(getattr(packet,k)) for k in ('children','parents','fast_coordinates','ema_coordinates',
                            'parent_row_state','parent_history','fast_features','ema_features')},
                        packet_fingerprint=packet.fingerprint,preview_context=captured['preview_context'],
                        models=cpu({name:getattr(trainer,name).state_dict() for name in ('G','D','ema_G')}))
                    torch.save(archive, case_dir / 'prepared-packet.pt')
                    # Independent predicates on the exact prepared features and
                    # current original rows, with no new draw or table mutation.
                    original_fast = mt.View(None,**captured['preview_context']['fast'])
                    original_ema = mt.View(None,**captured['preview_context']['ema'])
                    actual_fast = mt.observe_view(snapshot,packet.fast_features,coordinates=packet.fast_coordinates)
                    actual_ema = mt.observe_view(snapshot,packet.ema_features,coordinates=packet.ema_coordinates)
                    child_host,parent_host=packet.children.cpu(),packet.parents.cpu()
                    count=len(child_host)
                    retained=(original_fast.eligible[child_host]&original_fast.eligible[parent_host]
                        &original_ema.eligible[child_host]&original_ema.eligible[parent_host]
                        &actual_fast.eligible&actual_ema.eligible
                        &(original_fast.cells[child_host]==original_fast.cells[parent_host])
                        &(original_ema.cells[child_host]==original_ema.cells[parent_host])
                        &(actual_fast.categories==original_fast.categories[child_host])
                        &(actual_ema.categories==original_ema.categories[child_host])
                        &(actual_fast.groups==original_fast.groups[child_host])
                        &(actual_ema.groups==original_ema.groups[child_host])
                        &(actual_fast.groups==actual_ema.groups))
                    captured['packet_retention_exact']=bool(retained.all())
                    if not captured['packet_retention_exact']:
                        raise AssertionError('prepared copy lost both-view inside/category/group/support conditions')
                    stream_before = context.stream.get_state().clone()
                    result = original_commit(packet, context, snapshot)
                    if not torch.equal(stream_before,context.stream.get_state()):
                        raise AssertionError('commit drew noise')
                    if not torch.equal(context.prior.z[packet.children], packet.fast_coordinates):
                        raise AssertionError('FAST coordinates differ from prepared packet')
                    if not torch.equal(context.ema_prior.z[packet.children], packet.ema_coordinates):
                        raise AssertionError('EMA coordinates differ from prepared packet')
                    captured['packet_fingerprint'] = packet.fingerprint
                    captured['packet_inheritance_exact'] = all(torch.equal(context.row_state[k][packet.children],v)
                        for k,v in packet.parent_row_state.items())
                    captured['packet_history_exact'] = packet.parent_history is None or torch.equal(context.history[packet.children],packet.parent_history)
                    return result

                fc.FeatureCellSnapshot.ordinary_transport = transport
                fc.run_mean_phase = mean_call
                mt.commit_packet = commit_call
                mt.preview_pairs = preview_call
                bd._move = copy_call
                try: event = bd.maybe_apply(trainer, trainer.last_output_sigma)
                finally:
                    fc.FeatureCellSnapshot.ordinary_transport = original_transport
                    fc.run_mean_phase = original_mean
                    mt.commit_packet = original_commit
                    mt.preview_pairs = original_preview
                    if move_owned[0]: bd._move = move_owned[1]
                    else: bd.__dict__.pop('_move',None)
                moved = cpu(bd.moved_rows) if bd.moved_rows is not None else torch.empty(0,dtype=torch.long)
                table_tester = trainer._table_tester()
                tester_before = cpu(table_tester.state_dict()) if table_tester is not None else None
                evidence_before = cpu(trainer.row_evidence.state_dict()) if trainer.row_evidence is not None else None
                if table_tester is not None:
                    original_rebase = table_tester.rebase
                    rebase_owned = ('rebase' in table_tester.__dict__,table_tester.__dict__.get('rebase'))
                    def rebase(params, rows):
                        hook_calls.append(dict(kind='rebase', rows=cpu(rows).tolist()))
                        return original_rebase(params, rows)
                    table_tester.rebase = rebase
                if trainer.row_evidence is not None:
                    original_reset = trainer.row_evidence.reset
                    reset_owned = ('reset' in trainer.row_evidence.__dict__,trainer.row_evidence.__dict__.get('reset'))
                    def reset(rows):
                        hook_calls.append(dict(kind='row_evidence_reset', rows=cpu(rows).tolist()))
                        return original_reset(rows)
                    trainer.row_evidence.reset = reset
                try: exec(hook_code,dict(self=trainer,event=event))
                finally:
                    if table_tester is not None:
                        if rebase_owned[0]: table_tester.rebase=rebase_owned[1]
                        else: table_tester.__dict__.pop('rebase',None)
                    if trainer.row_evidence is not None:
                        if reset_owned[0]: trainer.row_evidence.reset=reset_owned[1]
                        else: trainer.row_evidence.__dict__.pop('reset',None)
                trainer._serve_release()  # Stored state and checks are the genuine FAST training view.
                after = trainer._state_dict()
                bd.check_state(after['birth_death'])
                bd.check_paired_average_step(after['birth_death'],trainer.completed_steps)
                torch.save(after, case_dir / 'after.pt')
                ordinary = captured['ordinary']; novel = ordinary['novel_birth_plan']
                mean_child, mean_parent = captured['mean_children'],captured['mean_parents']
                empty = torch.empty(0,dtype=torch.long)
                copy_child = torch.cat([c for c,p in old_copies]) if old_copies else empty
                copy_parent = torch.cat([p for c,p in old_copies]) if old_copies else empty
                all_child = torch.cat((copy_child,novel['children'],mean_child))
                all_sources = torch.cat((copy_parent,novel['source_seed_rows'],mean_parent))
                wanted_hooks = [] if not event['moves'] or trainer.lr_settle is None else ['rebase']+(['row_evidence_reset'] if trainer.row_evidence is not None else [])
                checks = dict(fresh_backend9_no_old_state_load=after['birth_death']['backend_schema']==9 and before['completed_steps']==0 and after['completed_steps']==1,
                    common_actual_family=event['count_multiplicity']==3*event['cells']+3 and event['count_cutoff']==.05/(3*event['cells']+3),
                    ordinary_budget=event['ordinary_moves']<=math.floor(.05*bd.N)==event['ordinary_budget'],
                    phase_sum=event['ordinary_moves']==sum(event[k] for k in ('ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves','ordinary_novel_birth_moves','ordinary_mean_moves')),
                    complete_moves=event['moves']==event['ordinary_moves']+event['iso_moves']==len(all_child),
                    unique_children=len(all_child)==len(torch.unique(all_child)),
                    unique_sources=len(all_sources)==len(torch.unique(all_sources)),
                    no_source_overwritten=not bool(torch.isin(all_child,all_sources).any()),
                    mean_reserves_prefix=not bool(torch.isin(torch.cat((mean_child,mean_parent)),captured['reserved_rows']).any()),
                    moved_union=torch.equal(moved.sort().values,all_child.sort().values),
                    actual_hook_union=[h['kind'] for h in hook_calls]==wanted_hooks and all(sorted(h['rows'])==moved.sort().values.tolist() for h in hook_calls),
                    final_fresh_lease=event['paired_average']['snapshot']==bd.snapshot_serial and event['paired_average']['step']==1,
                    original_model_weights_unchanged=model_before=={name:state_hash(getattr(trainer,name).state_dict()) for name in model_before},
                    dedicated_training_streams_unchanged=all(torch.equal(getattr(trainer,name).get_state(),v) for name,v in streams_before.items()),
                    original_saved_tensors_unchanged=state_hash(source)==original_hash,
                    global_CPU_RNG_unchanged=torch.equal(global_before,torch.get_rng_state()))
                if len(moved) and trainer.row_evidence is not None:
                    moved_device=moved.to(device)
                    checks['moved_own_evidence_zero']=all(not bool(getattr(trainer.row_evidence,k)[moved_device].any())
                        for k in trainer.row_evidence._TENSORS)
                    checks['own_evidence_reset_count']=trainer.row_evidence.counters['resets']==evidence_before['counters']['resets']+len(moved)
                if len(moved) and table_tester is not None:
                    moved_device=moved.to(device)
                    checks['moved_population_participation_revoked']=not bool(table_tester.stationary_rows[moved_device].any())
                    checks['moved_unfinished_block_invalidated']=table_tester.invalid_block_rows is not None and bool(table_tester.invalid_block_rows[moved_device].all())
                    checks['moved_tester_anchor_current']=torch.equal(table_tester.anchor.view(bd.N,-1)[moved_device],trainer.prior.z[moved_device])
                if device.type=='cuda': checks['global_CUDA_RNG_unchanged']=torch.equal(cuda_before,torch.cuda.get_rng_state(device))
                if label=='grid':
                    checks.update(mean_has_authority_and_progress=event['mean_transport']['status']=='firing' and len(mean_child)>0 and
                        event['mean_transport']['objective_after_mean']<event['mean_transport']['objective_before_mean'],
                        exact_packet_inheritance=captured.get('packet_inheritance_exact',False) and captured.get('packet_history_exact',False),
                        both_views_actual_inside_category_group_support_retained=captured.get('packet_retention_exact',False))
                else: checks['conservative_toy_mean_veto']=event['mean_transport']['status']=='veto' and not len(mean_child) and event['mean_transport']['attempts']==0
                if not all(checks.values()): raise AssertionError([k for k,v in checks.items() if not v])
                json.dumps(bd.diagnostics(),allow_nan=False)
                trace = dict(schema=1,case=label,device=str(device),ordinary_children=ordinary['action_children'].tolist(),
                    ordinary_copy_children=captured['ordinary_copy_children'].tolist(),ordinary_copy_parents=captured['ordinary_copy_parents'].tolist(),
                    ordinary_kinds=ordinary['action_kinds'].tolist(),old_copy_calls=[dict(phase='ordinary' if torch.equal(c,captured['ordinary_copy_children']) else 'isolation',
                        children=c.tolist(),parents=p.tolist()) for c,p in old_copies],
                    novel_children=novel['children'].tolist(),novel_source_seeds=novel['source_seed_rows'].tolist(),
                    mean_children=mean_child.tolist(),mean_parents=mean_parent.tolist(),prefix_reserved=captured['reserved_rows'].tolist(),
                    earlier_ordinary=captured['earlier_ordinary'],moved_rows=moved.tolist(),caller_hooks=hook_calls,
                    packet_fingerprint=captured.get('packet_fingerprint'),preview_detail=captured.get('preview_detail'),checks=checks,mean_transport=event['mean_transport'])
                (case_dir/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
                torch.save(dict(tester_before=tester_before,tester_after=cpu(table_tester.state_dict()) if table_tester else None,
                    evidence_before=evidence_before,evidence_after=cpu(trainer.row_evidence.state_dict()) if trainer.row_evidence else None),case_dir/'hook-state.pt')
                provenance=dict(schema=1,case=label,device=str(device),construction='fresh native GANTrainer/backend9; no backend8/full-state load or relabel',
                    original_checkpoint=str(path),original_checkpoint_sha256=sha(path),package_root=str(args.package_root),seed=314159,
                    source_model_table_FIFO_raw_tensors=True,copied_raw_components=['G','D','prior','ema_G','ema_prior','prior optimizer row tensors/scalars','latent_history','latent_bandwidth','log_sigma when learnable'],
                    fresh_components=['controller except latent_bandwidth','settlers/population participation','RowEvidence','lineage','all dedicated device streams','backend counters/semantic stamps'],
                    fresh_evidence_limitation='RowEvidence and pair participation initially zero; caller reset/rebase masks, anchor and counters tested, not inherited historical participation',
                    no_optimizer_or_GAN_gradient_step=True,completed_steps_before=0,completed_steps_after=1,
                    input_binding=seal['source_and_input_sha256'],output_sha256={p.name:sha(p) for p in case_dir.iterdir() if p.is_file()})
                (case_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
                record=dict(case=label,status='PASS',rows=bd.N,cells=event['cells'],ordinary=event['ordinary_moves'],mean=event['ordinary_mean_moves'],isolation=event['iso_moves'],
                    witness=event['mean_transport'],checks=checks,output_sha256={str(p):sha(p) for p in case_dir.iterdir() if p.is_file()},elapsed_seconds=time.perf_counter()-started)
                records.append(record)
                log('fresh_reaction_pass',case=label,ordinary=record['ordinary'],mean=record['mean'],lower_bound=record['witness']['lower_bound'])
        verify()
        if not torch.equal(global_before,torch.get_rng_state()): raise AssertionError('global CPU RNG changed')
        if device.type=='cpu' and torch.cuda.is_initialized(): raise AssertionError('CPU contract initialized CUDA')
        result=dict(status='PASS',device=str(device),records=records,source_and_input_sha256=seal['source_and_input_sha256'],
            source_preseal_sha256=sha(HERE/'SOURCE-FROZEN-v2.json'),cuda_initialized=torch.cuda.is_initialized(),
            new_training_steps=0,new_optimizer_steps=0,new_quality_emissions=0,new_seed_experiments=0,quality_verdict=None,fixture_version=2,
            fixture_scope='one constructed new-law reaction per frozen raw grid/toy input; device-native original dedicated seeds, not historical or cross-device replay')
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        log('mechanics_complete',status='PASS',output=str(args.output))
    except Exception as error:
        args.output.write_text(json.dumps(dict(status='FAIL',error=repr(error),traceback=traceback.format_exc(),records=records),indent=2)+'\n')
        raise


if __name__=='__main__': main()
