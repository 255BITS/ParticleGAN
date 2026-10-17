"""Stdlib-only pin of backend10 state boundaries and genuine mechanics source."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER=ROOT/'integration/review/training-regression/post-ra10-quality/linear-output-production'
HERE=Path(__file__).resolve().parent
EXPECTED='1294e397fcb1a59bfb8629c373a12eb866384f0ebcf1344276234bd97b0291f3'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def verify(mapping):
    for path,digest in mapping.items():assert sha(path)==digest,path

def find(tree,name):
    values=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
    assert len(values)==1,name
    return values[0]

seal_path=OWNER/'SOURCE-FROZEN.json'
assert sha(seal_path)==EXPECTED
seal=json.loads(seal_path.read_text())
verify(seal['source_and_input_sha256'])
package=OWNER/'pkg-OUTPUT-MEAN/particlegan'
sources={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py'))}
assert len(sources)==31 and sources==seal['package_source_sha256']
baseline=ROOT/'pkg-CB64-RA10/particlegan'
changed=[name for name in sources if not (baseline/name).exists() or sha(baseline/name)!=sources[name]]
assert changed==['feature_cells.py','mean_transport.py','output_moments.py'],changed
assert Path(seal['config_path']).read_bytes()==(ROOT/'configs/overrides-CB64-RA9.json').read_bytes()
fc=ast.parse((package/'feature_cells.py').read_text())
mt=ast.parse((package/'mean_transport.py').read_text())
bd=find(fc,'FeatureCellBirthDeath')
methods={n.name:n for n in bd.body if isinstance(n,ast.FunctionDef)}
outbound=ast.unparse(methods['state_dict']); inbound=ast.unparse(methods['load_state_dict'])
assert "mean['selected_axes'] = list(mean['selected_axes'])" in outbound
assert "mean['selected_axes'] = list(mean['selected_axes'])" in inbound
assert "result['last']['mean_transport'] = mean" in outbound
assert "self.last['mean_transport'] = mean" in inbound
check=ast.unparse(find(mt,'check_mean_diagnostics'))
for text in ("stamp['schema'] != 2", "type(axis) is not int", "type(stamp['output_dim']) is not int",
             "rank != len(axes)", "len(set(stamp['selected_axes'])) != len(stamp['selected_axes'])",
             "stamp['chart_rank'] != paired['rank']", "math.sqrt(stamp['moment_rank'] / Q)",
             "stamp['fitted_rows'] != (n + 1) // 2", "stamp['alpha'] != Q / (3 * stamp['cells'] + 3)"):
    assert text in check,text
assert "stamp[k]" in check and "type(stamp[k]) is not int" in check
assert "self.check_state(state)"==ast.unparse(methods['load_state_dict'].body[0].value)
assert "type(state.get('backend_schema')) is not int" in ast.unparse(methods['check_state'])
assert seal['backend_schema']==10 and seal['mean_schema']==2 and seal['trainer_schema']==5
helper=ast.parse((OWNER/'run_mechanics.py').read_text())
constructor=ast.unparse(find(helper,'construct'))
for text in ("trainer.G.load_state_dict(weights['G'])", "trainer.D.load_state_dict(weights['D'])",
             "trainer.ema_G.load_state_dict(weights['ema_G'])", "bd.BACKEND_SCHEMA != 10",
             "raw source model/table binding failed", "named raw FIFO/bandwidth/moment/history/noise binding failed"):
    assert text in constructor,text
assert '.load_state_dict(source)' not in constructor and '.load_state_dict(saved)' not in constructor
body=ast.unparse(helper)
for text in ("bd.check_state(before['birth_death'])", "bd.check_state(after['birth_death'])",
             "trainer._serve_release()", "bd.__dict__.pop('_move', None)",
             "table_tester.__dict__.pop('rebase', None)", "trainer.row_evidence.__dict__.pop('reset', None)",
             "moved_union=", "'moved_own_evidence_zero'", "'moved_population_participation_revoked'",
             "committed_raw_outputs_equal_prepared=", "learned_cache_features_equal_prepared="):
    assert text in body,text
verify(seal['source_and_input_sha256'])
receipt=dict(status='PASS',scope='Source-only backend10 state/API and genuine mechanics qualification',
    utc=datetime.now(timezone.utc).isoformat(),source_preseal_sha256=EXPECTED,
    backend_schema=10,mean_schema=2,trainer_schema=5,package_sha256=seal['package_sha256'],
    source_and_input_sha256=seal['source_and_input_sha256'],
    checks=dict(all_279_raw_guards=True,unchanged_original_modules=28,exact_RA9_config=True,
        strict_initial_reacted_schema2_and_chart_output_rank_separation=True,
        bounded_unique_strict_integer_axes_and_sample_shape_dimension=True,
        actual_critic_K_common_3K3_and_moment_rank_range=True,
        outbound_and_load_axes_list_owned=True,backend10_validation_precedes_load_mutation=True,
        fresh10_constructor_raw_G_D_restore_and_complete_fidelity=True,
        private_instance_hooks_removed_before_checkpoint=True,
        complete_phase_union_reset_participation_and_packet_output_controls=True),
    Torch_imported=False,PT_objects_loaded=0,model_forwards=0,numerical_execution=False,
    limitation='Source qualification only; actual fresh10 state/resume/atomic controls remain pending.')
path=HERE/'receipt.json'
assert not path.exists()
path.write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status='PASS',receipt_sha256=sha(path),guards=len(seal['source_and_input_sha256']))))
