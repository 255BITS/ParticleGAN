"""Seal source and already finalized inputs without importing numerical code."""
import ast
from datetime import datetime, timezone
from bindings import (HERE, ROOT, PLAN, GRID_AUDIT, HEAD, AFFINE, STATE_HASH,
                      SCREEN, EXPECTED, NUMERIC_INPUTS, sha, read, write_new,
                      merge_maps, verify, function_node, ast_sha)


def main():
    if (HERE / 'SOURCE-FROZEN.json').exists():
        raise SystemExit('retain the existing seal; use a separate attempt for any correction')
    verify(EXPECTED)
    grid, plan = read(GRID_AUDIT / 'FROZEN.json'), read(PLAN / 'FROZEN.json')
    assert grid['status'] == 'PASS' and grid['quality_verdict'] == 'FAIL'
    assert grid['canonical_fixture_validity'] == 'VALID' and grid['watcher_exited']
    assert plan['status'] == 'PASS_SOURCE_ONLY' and not plan['numerical_execution']
    guards = merge_maps({str(p): d for p, d in EXPECTED.items()},
        grid['source_and_input_sha256'], grid['artifact_sha256'], grid['local_sha256'],
        plan['source_and_input_sha256'], plan['local_sha256'])
    assert all(str(p) in guards for p in NUMERIC_INPUTS)
    verify(guards)
    for filename in ('bindings.py', 'seal_source.py', 'run_decomposition.py', 'seal_result.py'):
        path = HERE / filename
        compile(path.read_text(), str(path), 'exec')  # Parse only; no module execution.
        guards[str(path)] = sha(path)
    for filename in ('PROTOCOL.md', 'FRAME-ADDENDUM.md'):
        guards[str(HERE / filename)] = sha(HERE / filename)
    pair_functions = {name: function_node(SCREEN, name)
                      for name in ('draw', 'draw_holdout', 'score_dir')}
    draw, holdout, save = pair_functions.values()
    returned = [node for node in ast.walk(draw) if isinstance(node, ast.Return)]
    assert len(returned) == 1 and ast.dump(returned[0].value) == ast.dump(
        ast.parse('(clean, noisy, sigma)', mode='eval').body)
    # These statements preserve row order. The original source hash is fixed.
    expected_noise = ast.parse('clean + sigma * torch.randn_like(clean) if sigma else clean', mode='eval').body
    assert any(isinstance(n, ast.Assign) and ast.dump(n.value) == ast.dump(expected_noise)
               for n in ast.walk(draw))
    expected_holdout = ast.parse("out['clean'][model], out['noisy'][model] = clean.cpu().numpy(), noisy.cpu().numpy()").body[0]
    assert any(ast.dump(n) == ast.dump(expected_holdout) for n in ast.walk(holdout))
    expected_save = ast.parse("np.savez_compressed(d / 'holdout_samples.npz', **arrays)").body[0]
    assert any(ast.dump(n) == ast.dump(expected_save) for n in ast.walk(save))
    proof = dict(status='PASS_SOURCE_ONLY', pairing='same original draw; no row permutation',
        functions={name: ast_sha(node) for name, node in pair_functions.items()},
        immutable_function_ast={
            'head_features': ast_sha(function_node(HEAD, 'head_features')),
            'raw_affine': ast_sha(function_node(AFFINE, 'raw', affine_only=True)),
            'tensor_state_hash': ast_sha(function_node(STATE_HASH, 'tensor_state_hash'))},
        paired_feature_frame='g(C) selects both group bin and center/scale/clip frame for C and Y',
        anchor_to_clean='unpaired empirical distribution difference',
        numerical_imports_PT_objects_models_charts_or_forwards=0)
    write_new(HERE / 'FUNCTION-AST.json', proof)
    guards[str(HERE / 'FUNCTION-AST.json')] = sha(HERE / 'FUNCTION-AST.json')
    verify(guards)
    seal = dict(status='FROZEN_PRE_NUMERICAL_EXECUTION', utc=datetime.now(timezone.utc).isoformat(),
        selected_design_sha256=sha(PLAN / 'DESIGN.md'),
        source_and_input_sha256=guards, raw_guarded_files=len(guards),
        numeric_inputs=[str(p) for p in NUMERIC_INPUTS],
        original_grid_fixture_status='VALID', original_grid_quality='FAIL',
        numerical_execution=False, Torch_imported=False, PT_objects_loaded=0,
        chart_fits=0, forwards=0, new_draws=0, actions=0, scoring_calls=0,
        execution_requires='independent helper review and explicit root authorization')
    write_new(HERE / 'SOURCE-FROZEN.json', seal)
    print('SOURCE_ONLY_FROZEN', len(guards), sha(HERE / 'SOURCE-FROZEN.json'), flush=True)


if __name__ == '__main__':
    main()
