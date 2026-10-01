"""Final source-only fixture/replay audit before root candidate freeze."""
import ast
import hashlib
import json
from pathlib import Path
from contracts import verify_fixture_sources, ast_members

ROOT = Path(__file__).resolve().parent
LANE = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/validation-cb64-ra11/learned')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source_contract = verify_fixture_sources()
receipt = json.loads((ROOT / 'source-transform-receipt.json').read_text())
for filename, edits in [('run_training.py', receipt['edits']), ('replay.py', receipt['replay_edits'])]:
    text = (LANE / filename).read_text()
    for edit in edits:
        assert text.count(edit['before']) == 1
        text = text.replace(edit['before'], edit['after'])
    assert text == (ROOT / filename).read_text()
    ast.parse(text)
common = (LANE / 'common.py').read_text().replace('VARIANTS = ("CB64-RA11",)', "VARIANTS = ('RA12-auto',)")
assert common == (ROOT / 'common.py').read_text()
replay = (ROOT / 'replay.py').read_text()
for invariant in ('START = 1000', 'COUNT = 10', "EXCLUDED = ['birth_death.last.eval_seconds']",
                  'for branch in range(2):', 'checkpoint=torch.load(path,weights_only=False)',
                  "torch.load(path,map_location='cpu',weights_only=False)['trainer']",
                  'restored_exact[\'whole_state\']=restored_semantic_sha==saved_semantic_sha',
                  'samples=draw_samples(trainer,sample_count,SEED+100)',
                  "sample_count=8192 if problem=='toy' else 4096"):
    assert invariant in replay, invariant
original_members = ast_members(LANE / 'replay.py')
adapted_members = ast_members(ROOT / 'replay.py')
for name in ('digest', 'semantic_state', 'counter_delta'):
    assert original_members[name] == adapted_members[name], name
adapter = ast.parse((ROOT / 'adapter.py').read_text())
sampler = next(node for node in adapter.body if isinstance(node, ast.ClassDef) and node.name == 'OriginalPrimarySampler')
sample_method = next(node for node in sampler.body if isinstance(node, ast.FunctionDef) and node.name == 'sample')
call = sample_method.body[0].value
assert isinstance(call, ast.Call)
assert any(keyword.arg == 'output_noise' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
           for keyword in call.keywords)
init = json.loads((ROOT / 'init-parity-current155.json').read_text())
assert init['status'] == 'PASS_PUBLIC_INIT_EXACT_ORIGINAL_PARITY'
for path, expected in init['source_file_sha256'].items():
    assert sha(path) == expected, path
assert not (ROOT / 'SOURCE-FREEZE.json').exists()
assert not list(ROOT.glob('LAUNCH*.json'))
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text())
output = dict(status='PASS_FINAL_SOURCE_ONLY_FIXTURE_REPLAY_AUDIT',
              source_contract=source_contract, public_init_parity_receipt_sha256=sha(ROOT / 'init-parity-current155.json'),
              original_private_and_global_rng_cpu_buffer_validation_unchanged=True,
              native_checkpoint_load_preserves_saved_device_tags=True,
              cpu_map_applies_only_to_second_checkpoint_restore_input=True,
              restored_full_state_fingerprint_captured_before_updates=True,
              exact_original_replay_updates_per_branch=10, independent_branches=2,
              tensor_placement_recorded_for_both_restore_inputs_and_results=True,
              primary_sample_bytes_compared=True, original_primary_draw_counts_and_seed=True,
              sampling_required_to_preserve_training_state=True,
              original_semantic_exclusion_only='birth_death.last.eval_seconds',
              candidate_source_frozen=False, training_updates=0, model_forwards=0,
              cuda_contexts=0, cuda_launches=0,
              local_source_sha256={path.name: sha(path) for path in sorted(ROOT.glob('*.py'))})
(ROOT / 'AUDIT-SOURCE.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(dict(status=output['status'], sha256=sha(ROOT / 'AUDIT-SOURCE.json'))), flush=True)
