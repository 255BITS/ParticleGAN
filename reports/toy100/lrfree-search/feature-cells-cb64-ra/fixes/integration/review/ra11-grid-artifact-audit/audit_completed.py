"""Completed RA11 grid saved-artifact audit; CPU storage reads, no forwards."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import zipfile

sys.dont_write_bytecode = True
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
LANE = ROOT / 'validation-cb64-ra11'
PACKAGE = ROOT / 'pkg-CB64-RA11'
AUTHORITY = ROOT / 'integration/review/audit_learned.py'
NATIVE_AUTHORITY = ROOT.parent / 'feature-cells-cuda-retest-20260929/audit/check_artifacts.py'
STEPS = [0, 1, 10, 25, 50, 100] + list(range(250, 7001, 250))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def verify(mapping):
    for p, h in mapping.items():
        assert sha(p) == h, p


def nodes(path, names, cls=None):
    tree = ast.parse(Path(path).read_text())
    body = tree.body if cls is None else next(n.body for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == cls)
    found = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in found} == set(names)
    return found


def execute(definitions, namespace, label):
    exec(compile(ast.Module(body=definitions, type_ignores=[]), label, 'exec'), namespace)
    return namespace


def rendered_length(value, *, kinds=False, n):
    if isinstance(value,list):
        assert len(value)<=32
        if kinds:
            assert all(type(row) is int and row in (0,1,2,3,4) for row in value)
        else:
            assert all(type(row) is int and 0<=row<n for row in value)
            assert len(value)==len(set(value))
        return len(value)
    assert type(value) is str and value.startswith('<list len=') and value.endswith('>')
    length=int(value[10:-1])
    assert length>32 and value==f'<list len={length}>' and length<=n
    return length


def check_rendered_actions(last,serial,n):
    names=('ordinary_mean_children','ordinary_mean_parents','ordinary_action_children',
           'ordinary_copy_parent_rows','ordinary_action_kinds')
    sizes={k:rendered_length(last[k],kinds=k=='ordinary_action_kinds',n=n) for k in names}
    assert sizes['ordinary_mean_children']==sizes['ordinary_mean_parents']
    available=all(isinstance(last[k],list) for k in names)
    if serial:
        fields=('ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves','ordinary_mean_moves',
                'ordinary_copy_moves','ordinary_novel_birth_moves','ordinary_moves','iso_moves','moves')
        assert all(type(last[k]) is int and last[k]>=0 for k in fields)
        assert last['ordinary_copy_moves']==sum(last[k] for k in fields[:4])
        assert last['ordinary_moves']==last['ordinary_copy_moves']+last['ordinary_novel_birth_moves']<=math.floor(.05*n)
        assert last['moves']==last['ordinary_moves']+last['iso_moves']
        assert sizes['ordinary_mean_children']==last['ordinary_mean_moves']
        assert sizes['ordinary_action_children']==sizes['ordinary_action_kinds']==last['ordinary_moves']
        assert sizes['ordinary_copy_parent_rows']==last['ordinary_copy_moves']
        if isinstance(last['ordinary_action_kinds'],list):
            for kind,key in enumerate(('ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves',
                                      'ordinary_novel_birth_moves','ordinary_mean_moves')):
                assert last['ordinary_action_kinds'].count(kind)==last[key]
    else:
        assert all(s==0 for s in sizes.values())
    return dict(all_row_lists_available=available,rendered_lengths=sizes,
        historical_compressed_row_IDs_not_reconstructed=not available)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    seal = read(args.input_freeze)
    assert seal['status'] == 'COMPLETED_GRID_INPUTS_FROZEN' and seal['steps'] == STEPS
    verify(seal['source_and_input_sha256'])  # All raw guards before Torch or PT.
    assert args.output.resolve().is_relative_to(HERE) and not args.output.exists()
    args.output.mkdir()
    run = LANE / 'screens/runs/grid100'
    result = read(run / 'result.json')
    canonical = read(args.input_freeze.parent / 'canonical-grid-acceptance-receipt.json')
    assert result['status'] in ('PASS', 'FAIL') and canonical['canonical_fixture_validity'] == 'VALID'
    assert result['completed_steps'] == 7000 and result['observations'] == len(STEPS)

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = torch.get_rng_state().clone()
    authority = execute(nodes(AUTHORITY, ('read', 'sha', 'require', 'verify_sources',
        'load_cpu', 'digest', 'semantic', 'metadata', 'rng_placement')),
        dict(torch=torch, hashlib=hashlib, json=json, Path=Path, struct=struct,
            args=SimpleNamespace(validation=LANE), captured_freeze=None), '<unchanged-original-artifact-authority>')
    integrity = authority['verify_sources']()
    saved, devices = authority['load_cpu'](run / 'final-state.pt')
    state = saved['trainer']
    required = next(node.value for node in ast.parse(NATIVE_AUTHORITY.read_text()).body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REQUIRED_STATE' for t in node.targets))
    assert ast.literal_eval(required).issubset(state)
    whole_digest = authority['digest'](state, devices)
    semantic_digest = authority['digest'](authority['semantic'](state), devices)
    placement = authority['rng_placement'](state, devices)

    native = execute(nodes(NATIVE_AUTHORITY, ('Checks', 'read', 'lines', 'npz_headers', 'cloud', 'native')),
        dict(ast=ast, json=json, Path=Path, zipfile=zipfile,
            HARNESS=Path('/ml2/hypergan/lrfree-20260926/harness'), NATIVE_STEPS=STEPS), '<unchanged-original-native-authority>')
    original_checks = native['Checks']()
    native['native'](original_checks, 'grid100', run, result)
    assert all(c['ok'] for c in original_checks.rows), [c for c in original_checks.rows if not c['ok']]
    (args.output / 'original-native-authority.json').write_text(json.dumps(original_checks.rows, indent=2) + '\n')

    renderer = execute(nodes(Path('/ml2/hypergan/lrfree-20260926/harness/screen.py'), ('finite','jsonable')),
        dict(torch=torch,math=math), '<unchanged-original-native-json-renderer>')

    feature = ast.parse((PACKAGE / 'particlegan/feature_cells.py').read_text())
    resolution = [n for n in feature.body if (isinstance(n, ast.FunctionDef) and n.name == '_fit_cell_count')
        or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CELL_RESOLUTION_POLICY' for t in n.targets))]
    methods = execute(resolution + nodes(PACKAGE / 'particlegan/feature_cells.py',
        ('_check_paired_average_state', 'check_paired_average_step', 'paired_average_eligible', '_check_mean_actions'),
        cls='FeatureCellBirthDeath'), dict(torch=torch, math=math, Fraction=Fraction, Q=.05), '<exact-backend10-state-predicates>')
    output_tree=ast.parse((PACKAGE/'particlegan/output_moments.py').read_text())
    output_assigns=[node for node in output_tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name)
        and t.id in {'Q','POLICY','PROJECTION_POLICY','FIT_CHUNK','MAX_RANK'} for t in node.targets)]
    output_constants=execute(output_assigns,{},'<exact-output-frame-constants>')
    mean_tree = ast.parse((PACKAGE / 'particlegan/mean_transport.py').read_text())
    assignments = [n for n in mean_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
        and t.id in ('Q', 'POLICY', 'MEAN_POLICY', 'PLANNING_POLICY', 'INSIDE_POLICY', 'SCALAR_FIELDS', 'MEAN_KEYS')
        for t in n.targets)]
    mean = execute(assignments + nodes(PACKAGE / 'particlegan/mean_transport.py',
        ('initial_mean_diagnostics', 'check_mean_diagnostics')), dict(math=math,OUTPUT_MEAN_POLICY=output_constants['POLICY'],PROJECTION_POLICY=output_constants['PROJECTION_POLICY'],MAX_RANK=output_constants['MAX_RANK']), '<exact-mean-scalar-predicates>')
    bd = state['birth_death']
    settings = bd['settings']
    assert type(state['schema']) is int and state['schema'] == 5
    assert state['completed_steps'] == 7000 and state['device'] == 'cuda:0' and state['serial_backward'] is True
    assert type(bd['backend_schema']) is int and bd['backend_schema'] == 10 and bd['backend'] == 'feature_cells'
    n = len(state['models']['prior']['z'])
    assert n == 20000 and state['recipe']['z_dim'] == 2 and state['recipe']['batch_size'] == 2048
    assert settings['cells'] == state['recipe']['birth_death_cells'] == 128
    assert settings['resolution_policy'] == methods['CELL_RESOLUTION_POLICY']
    assert settings['mean_policy'] == mean['MEAN_POLICY'] and settings['mean_planning_policy'] == mean['PLANNING_POLICY']
    assert settings['mean_inside_policy'] == mean['INSIDE_POLICY'] and settings['mean_rank_bound'] == 8 and settings['mean_pool'] == 64
    assert settings['mean_projection_policy']==output_constants['PROJECTION_POLICY'] and settings['mean_frame_fit_chunk']==output_constants['FIT_CHUNK']
    assert settings['count_family'] == 'original_K_plus_support_2K_plus_global_2_plus_mean_1_common_Q_over_3K_plus_3'
    assert settings['novel_birth_policy'] == 'paired_even_real_anchor_shared_3K_plus_3_v1'
    assert not any(k in bd for k in ('snapshot', 'latent_geometry', 'moved_rows', 'mean_packet', 'fixed_moment','output_projection','output_moment_frame'))
    assert all(p.device.type == 'cpu' for model in state['models'].values() for p in model.values())
    assert json.loads(json.dumps(state['recipe'],default=str)) == result['recipe']
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line]
    assert [row['step'] for row in rows] == STEPS
    checked_rows = []
    previous = None
    for row in rows:
        diag = row['diag']['birth_death']
        last, stamp, serial = diag['last'], diag['paired_average'], diag['snapshot_serial']
        assert diag['backend'] == 'feature_cells' and diag['settings'] == settings
        view = dict(last=last, paired_average=stamp, snapshot_serial=serial)
        fake = SimpleNamespace(N=n, settings=settings, paired_average=stamp, snapshot_serial=serial,
            snapshot=None, fill=min(n,row['step']*2048), rows_since_eval=diag['paired_average_age_real_rows'], dry_run=False)
        methods['_check_paired_average_state'](fake, view)
        methods['check_paired_average_step'](fake, view, row['step'])
        action_scope = check_rendered_actions(last, serial, n)
        if action_scope['all_row_lists_available']:
            methods['_check_mean_actions'](fake, view)
        mean['check_mean_diagnostics'](last['mean_transport'], last=last, paired=stamp,
            n=n, snapshot_serial=serial, dry_run=False,sample_shape=bd['sample_shape'])
        count = diag['counters']
        assert all(type(value) is int and value >= 0 for value in count.values())
        assert count['mean_evals'] == serial == count['evals']
        assert 0 <= count['mean_witness_fires'] <= serial and count['mean_moves'] >= last['mean_transport']['moves']
        assert count['ordinary_moves'] == count['moves'] and count['matched'] == count['moves'] - count['novel_birth_moves']
        assert type(diag['paired_average_age_real_rows']) is int and diag['paired_average_age_real_rows'] >= 0
        if previous is not None:
            assert serial >= previous['snapshot_serial']
            assert count.keys() == previous['counters'].keys()
            assert all(count[k] >= previous['counters'][k] for k in count)
            if serial == previous['snapshot_serial']:
                assert last == previous['last'] and stamp == previous['paired_average']
        previous = diag
        checked_rows.append(dict(step=row['step'], snapshot=serial, reaction_step=stamp['step'],
            mean_status=last['mean_transport']['status'], mean_moves=last['mean_transport']['moves'],
            ordinary_moves=last.get('ordinary_moves',0), isolation_moves=last.get('iso_moves',0),
            mean_cumulative=count['mean_moves'], lease_eligible=stamp['eligible'],
            lease_age_real_rows=diag['paired_average_age_real_rows'], action_scope=action_scope))
    assert previous['last'] == renderer['jsonable'](torch,bd['last'],depth=1) and previous['counters'] == bd['counters']
    assert previous['paired_average'] == bd['paired_average'] and previous['snapshot_serial'] == bd['snapshot_serial']
    assert previous['paired_average_age_real_rows'] == bd['rows_since_eval']
    assert previous['population_policy'] == bd['population_policy']
    fake = SimpleNamespace(N=n, settings=settings, paired_average=bd['paired_average'], snapshot_serial=bd['snapshot_serial'],
        snapshot=None, fill=bd['fill'], rows_since_eval=bd['rows_since_eval'], dry_run=False)
    methods['_check_paired_average_state'](fake, bd)
    methods['check_paired_average_step'](fake, bd, state['completed_steps'])
    methods['_check_mean_actions'](fake, bd)
    mean['check_mean_diagnostics'](bd['last']['mean_transport'],last=bd['last'],paired=bd['paired_average'],n=n,snapshot_serial=bd['snapshot_serial'],dry_run=False,sample_shape=bd['sample_shape'])
    eligible = methods['paired_average_eligible'](fake, state['completed_steps'])
    assert 0 <= bd['fill'] <= n and 0 <= bd['cursor'] < n and bd['rows_since_eval'] >= 0
    assert bd['sample_shape'] == (2,) or bd['sample_shape'] == [2]
    assert bd['reservoir'].shape == (n,2) and bool(torch.isfinite(bd['reservoir']).all())
    for model in state['models'].values():
        for value in model.values():
            if value.is_floating_point(): assert bool(torch.isfinite(value).all())
    graph = bd['lineage_neighbors']
    source_rows = torch.arange(n)[:,None].expand_as(graph)
    edges = graph >= 0
    assert graph.shape == (n,8) and graph.dtype == torch.long
    assert bool(((graph >= -1) & (graph < n)).all()) and not bool(((graph == source_rows) & edges).any())
    pairs = (source_rows[edges]*n + graph[edges]).sort().values
    inverse = (graph[edges]*n + source_rows[edges]).sort().values
    assert torch.equal(pairs,inverse) and len(torch.unique(pairs)) == len(pairs)
    assert previous['lineage_edges'] == len(pairs)//2
    evidence = state['row_evidence']
    assert evidence['counters']['resets'] == bd['counters']['moves'] + bd['counters']['iso_moves']
    assert evidence['counters']['updates'] == 7000
    table = state['lr_settle'][0][1]
    assert table['population_schema'] == 1 and table['population_q'] == .05
    assert table['population_policy'] == 'two_pair_participation_Q_survival_one_descent_undo_v1'
    mask = table['stationary_rows']
    assert mask.shape == (n,) and mask.dtype == torch.bool and table['population_active'] == (table['last_decisive'] == -1)
    if table['population_active']:
        assert int(mask.sum()) >= n-math.floor(.05*n) and table['stationary_undo_s'] == table['s']/.5
    else: assert table['stationary_undo_s'] is None
    boundary = None
    if bd['snapshot_serial'] and bd['last']['step'] == state['completed_steps']:
        last = bd['last']
        children, mc, mp = last['ordinary_action_children'], last['ordinary_mean_children'], last['ordinary_mean_parents']
        group = state['optimizers'][0]['param_groups'][1]
        moments = state['optimizers'][0]['state'].get(group['params'][0],{})
        for value in moments.values():
            if isinstance(value,torch.Tensor) and value.shape == state['models']['prior']['z'].shape:
                assert torch.equal(value[mc],value[mp])
        history = state['optimizers'][0]['regularizer']['latent']['history']
        assert torch.equal(history[mc],history[mp])
        for key in ('M','Qs','W','S','flag'): assert not bool(evidence[key][children].any())
        if children:
            assert bool(table['invalid_block_rows'][children].all())
            if table['population_active']: assert not bool(mask[children].any())
        if mc:
            assert bool((graph[torch.tensor(mc)] == torch.tensor(mp)[:,None]).any(1).all())
        boundary = dict(ordinary_rows=len(children), mean_rows=len(mc), inherited_parent_moments_history=True,
            own_evidence_reset=True, unfinished_participation_invalidated=True, live_mean_lineage_links=True)
    verify(seal['source_and_input_sha256'])
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    receipt = dict(status='VALID', evidence_status='VALID', utc=datetime.now(timezone.utc).isoformat(),
        package_sha256=seal['package_sha256'], config_sha256=seal['config_sha256'],
        source_integrity=integrity, source_and_input_sha256=seal['source_and_input_sha256'],
        original_native_authority_checks=len(original_checks.rows), original_function_ASTs_unchanged=True,
        original_native_checks_executed_once_on_this_completed_input=True,
        final_state_sha256=sha(run/'final-state.pt'), whole_state_gpu_typed_sha256=whole_digest,
        semantic_state_gpu_typed_sha256=semantic_digest, RNG_placement=placement,
        trainer_schema=5, backend_schema=10, observations=checked_rows,
        final_mean=bd['last']['mean_transport'], final_counters=bd['counters'],
        final_lease=bd['paired_average'], final_lease_age_real_rows=bd['rows_since_eval'], final_lease_eligible=eligible,
        final_lineage_edges=len(pairs)//2, reaction_boundary_evidence=boundary,
        quality_verdict=result['status'], canonical_fixture_validity=canonical['canonical_fixture_validity'],
        new_scoring_calls=0, model_constructions=0, model_forwards=0, new_training_updates=0,
        new_optimizer_updates=0, new_quality_emissions=0, new_seeds=0, CPU_only=True, cuda_initialized=False,
        limits=['Only final-state.pt is saved; earlier 34 observations provide metadata, not historical tensor endpoints.',
            'Original native JSON replaces lists longer than 32 with length strings. Historical compressed IDs are unavailable; only rendered lengths and scalar balances are validated.',
            'Own row reset/moment/history/link predicates apply only at an exact saved reaction boundary.',
            'Isolation row IDs and historical unsaved population participants are unavailable; cumulative reset totals are checked.',
            'No new scorer/sample/trajectory or historical/cross-device replay. Audit validity is separate from original quality PASS/FAIL.'])
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='VALID',observations=len(checked_rows),quality_verdict=result['status'],cuda_initialized=False)),flush=True)


if __name__ == '__main__': main()
