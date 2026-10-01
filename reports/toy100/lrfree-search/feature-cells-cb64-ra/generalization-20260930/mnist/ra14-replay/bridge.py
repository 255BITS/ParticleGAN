"""Pinned restoration-only compatibility with the closed RA13 learned lane."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERAL = ROOT.parents[1]
ORIGIN = ROOT.parent / 'ra13-settled'
OLD_NAME = 'RA13-settled'
NAME = 'RA14-replay'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def package_sources(root):
    package = Path(root) / 'particlegan'
    return {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*.py'))}


def package_digest(root):
    digest = hashlib.sha256()
    package = Path(root) / 'particlegan'
    for path in sorted(package.rglob('*.py')):
        digest.update(str(path.relative_to(package)).encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def without_helper(path):
    source = Path(path).read_text()
    lines = source.splitlines(keepends=True)
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == '_state_to_device']
    assert len(nodes) == 1 and not nodes[0].decorator_list
    node = nodes[0]
    return ''.join(lines[:node.lineno - 1] + lines[node.end_lineno:])


def restoration_only_proof():
    old = GENERAL / 'pkg-RA13-settled'
    new = GENERAL / 'pkg-RA14-replay'
    assert (GENERAL / 'RA14-SOURCE-CLOSURE.json').is_file(), 'RA14 source is not closed'
    old_sources, new_sources = package_sources(old), package_sources(new)
    assert old_sources.keys() == new_sources.keys(), 'package source inventory changed'
    changed = [name for name in old_sources if old_sources[name] != new_sources[name]]
    assert changed == ['policy.py'], changed
    assert without_helper(old / 'particlegan/policy.py') == without_helper(new / 'particlegan/policy.py')
    old_config = GENERAL / 'configs/RA13-settled.json'
    new_config = GENERAL / 'configs/RA14-replay.json'
    assert old_config.read_bytes() == new_config.read_bytes(), 'config bytes changed'
    calls = []
    private_load_calls = []
    for path in sorted((new / 'particlegan').rglob('*.py')):
        def visit(node, scope=()):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope = (*scope, node.name)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == '_state_to_device':
                    calls.append(dict(path=str(path.relative_to(new / 'particlegan')),
                                      line=node.lineno, scope='.'.join(scope)))
                if isinstance(node.func, ast.Attribute) and node.func.attr in ('_check_state', '_load_state_dict'):
                    private_load_calls.append(dict(path=str(path.relative_to(new / 'particlegan')),
                                                   line=node.lineno, called=node.func.attr, scope='.'.join(scope)))
            for child in ast.iter_child_nodes(node):
                visit(child, scope)
        visit(ast.parse(path.read_text()))
    allowed = {'_state_to_device', 'UpdatePolicy._check_state',
               'UpdatePolicy.load_state_dict', 'GANTrainer._load_state_dict'}
    assert all(item['scope'] in allowed for item in calls), calls
    assert all(item['scope'] in {'UpdatePolicy.load_state_dict', 'GANTrainer.load_state_dict'}
               for item in private_load_calls), private_load_calls
    return dict(status='PASS_RESTORATION_ONLY_SOURCE_COMPATIBILITY',
                origin_package_root=str(old), origin_package_sha256=package_digest(old),
                candidate_package_root=str(new), candidate_package_sha256=package_digest(new),
                origin_source_sha256=old_sources, candidate_source_sha256=new_sources,
                changed_sources=changed, only_changed_function='policy._state_to_device',
                nonhelper_source_bytes_identical=True, helper_calls=calls,
                private_load_calls=private_load_calls, config_bytes_identical=True,
                origin_config_sha256=sha(old_config), candidate_config_sha256=sha(new_config),
                fresh_training_arithmetic_identical=True,
                proof_scope='Only checkpoint restoration and validation change. This does not label old training as newly executed.',
                source_closure_sha256=sha(GENERAL / 'RA14-SOURCE-CLOSURE.json'))


def build_bridge():
    assert not (ROOT / 'SOURCE-FREEZE.json').exists()
    assert not (ROOT / 'RESTORATION-BRIDGE.json').exists()
    proof = restoration_only_proof()
    closed = read(ORIGIN / 'CLOSED.json')
    assert closed['status'] == 'COMPLETE_ORIGINAL_TRAINING_RECOVERED_SOURCE_VALID_REPLAY_FAIL'
    assert closed['original_toy_quality_gate'] == 'PASS'
    assert closed['mnist_all_checkpoint_metrics_equal_corrected_E22']
    assert closed['mnist_all_checkpoint_lrs_equal_corrected_E22']
    assert closed['toy_all_postupdate_metrics_and_lrs_equal_original_RA11']
    for name, expected in closed['file_sha256'].items():
        assert sha(ORIGIN / name) == expected, name
    inputs = read(ORIGIN / 'INPUTS.json')
    package = inputs['variants'][OLD_NAME]
    assert package['package_sha256'] == proof['origin_package_sha256']
    assert package['source_sha256'] == proof['origin_source_sha256']
    assert package['config_sha256'] == proof['origin_config_sha256']
    pins = {str(ORIGIN / name): sha(ORIGIN / name)
            for name in ('CLOSED.json', 'INPUTS.json', 'SOURCE-FREEZE.json', 'REPORT.md', 'comparison.json')}
    fixtures = {}
    for problem in ('toy', 'mnist'):
        folder = ORIGIN / 'training' / problem / OLD_NAME
        config_path = folder / 'config.json'
        checkpoint = folder / 'checkpoint-1000.pt'
        result_path = folder / 'result.json'
        replay_path = ORIGIN / 'replay' / problem / OLD_NAME / 'result.json'
        result, run, replay = read(result_path), read(config_path), read(replay_path)
        assert result['status'] == 'COMPLETE' and result['steps'] == 2000
        assert run['package'] == package
        assert run['source_freeze_sha256'] == sha(ORIGIN / 'SOURCE-FREEZE.json')
        assert replay['start_step'] == 1000 and replay['steps_replayed'] == 10
        assert replay['restoration_semantic_bit_identical'] and replay['losses_bit_identical']
        native = replay['branches'][0]
        assert native['checkpoint_load_map_location'] == 'native'
        assert all(native['restored_semantic_sections_identical'].values())
        for path in (config_path, checkpoint, result_path, replay_path, Path(native['endpoint'])):
            pins[str(path)] = sha(path)
        fixtures[problem] = dict(checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                                 run_config_path=str(config_path), run_config_sha256=sha(config_path),
                                 source_freeze_sha256=sha(ORIGIN / 'SOURCE-FREEZE.json'), package=package,
                                 native_control_result=str(replay_path), native_control_endpoint=native['endpoint'])
    bridge = dict(status='SOURCE_COMPATIBLE_ORIGINAL_CHECKPOINT_BRIDGE',
                  prepared_utc=datetime.now(timezone.utc).isoformat(), origin_lane=str(ORIGIN),
                  origin_variant=OLD_NAME, candidate_variant=NAME, proof=proof, fixtures=fixtures,
                  original_updates_per_branch=10, original_branches_per_fixture=2,
                  additional_training_updates=0, numerical_results_inherited_by_source_proof=True,
                  fresh_numerical_scope='RA14 checkpoint restoration and original continuation only',
                  read_only_file_sha256=pins)
    (ROOT / 'RESTORATION-BRIDGE.json').write_text(json.dumps(bridge, indent=2) + '\n')
    return bridge


def checkpoint_origin(inputs, problem, variant):
    bridge = inputs['restoration_bridge']
    assert variant == NAME == bridge['candidate_variant']
    assert bridge['proof']['candidate_package_sha256'] == inputs['variants'][variant]['package_sha256']
    return bridge['fixtures'][problem]


def compare_native_control(output, inputs, problem):
    origin = checkpoint_origin(inputs, problem, NAME)
    old = read(origin['native_control_result'])['branches'][0]
    updates = [dict(step=new['step'], loss_bits=new['loss_sha256'] == saved['loss_sha256'],
                    semantic_state_bits=new['semantic_state_sha256'] == saved['semantic_state_sha256'],
                    semantic_section_bits=new['semantic_sections'] == saved['semantic_sections'])
               for new, saved in zip(output['update_fingerprints'], old['update_fingerprints'])]
    assert [item['step'] for item in updates] == list(range(1001, 1011))
    checks = dict(restored_semantic_state_bits=output['restored_state_semantic_sha256'] == old['restored_state_semantic_sha256'],
                  loss_bits=output['losses_sha256'] == old['losses_sha256'],
                  endpoint_semantic_state_bits=output['semantic_state_sha256'] == old['semantic_state_sha256'],
                  endpoint_semantic_section_bits=output['semantic_sections'] == old['semantic_sections'],
                  primary_sample_bytes=output['primary_samples_sha256'] == old['primary_samples_sha256'])
    passed = all(checks.values()) and all(all(v for k, v in item.items() if k != 'step') for item in updates)
    return dict(status='PASS' if passed else 'FAIL', origin_result=origin['native_control_result'],
                origin_result_sha256=sha(origin['native_control_result']), checks=checks, per_update=updates)


def preflight_bridge(inputs, torch):
    bridge = inputs['restoration_bridge']
    assert bridge == read(ROOT / 'RESTORATION-BRIDGE.json')
    assert bridge['proof'] == restoration_only_proof()
    before = torch.get_rng_state().clone()
    assert not torch.cuda.is_initialized()
    fixtures = {}
    for problem in ('toy', 'mnist'):
        origin = checkpoint_origin(inputs, problem, NAME)
        checkpoint = torch.load(origin['checkpoint_path'], map_location='cpu', weights_only=False)
        state = checkpoint['trainer']
        assert state['completed_steps'] == 1000 and state['device'] == 'cuda:0'
        assert checkpoint['data_position'] == 2 * 1000 * 128
        assert checkpoint['receipt_sha256'] == origin['run_config_sha256']
        table = state['lr_settle'][0][1]
        assert table['last_block'] is table['blocks'][-1]
        assert table['last_block'].untyped_storage().data_ptr() == table['blocks'][-1].untyped_storage().data_ptr()
        assert table['last_block'].shape == (1024 * 128,)
        assert all(state['recipe'][key] == value for key, value in inputs['variants'][NAME]['config'].items())
        assert state['recipe']['z_dim'] == 128 and state['recipe']['num_particles'] == 1024 and state['recipe']['batch_size'] == 128
        fixtures[problem] = dict(checkpoint_sha256=sha(origin['checkpoint_path']), completed_steps=1000,
                                real_stream_cursor=checkpoint['data_position'], source_recipe_matches_candidate=True,
                                original_table_displacement_alias_preserved_on_cpu_read=True)
    assert torch.equal(torch.get_rng_state(), before) and not torch.cuda.is_initialized()
    return dict(status='PASS_ZERO_UPDATE_SOURCE_CHECKPOINT_BRIDGE', fixtures=fixtures,
                training_updates=0, begin_step_calls=0, model_forwards=0, sampling_calls=0,
                global_rng_unchanged=True, cuda_context_initialized=False,
                restoration_bridge_sha256=sha(ROOT / 'RESTORATION-BRIDGE.json'))


if __name__ == '__main__':
    bridge = build_bridge()
    print(json.dumps(dict(status=bridge['status'], bridge_sha256=sha(ROOT / 'RESTORATION-BRIDGE.json'))))
