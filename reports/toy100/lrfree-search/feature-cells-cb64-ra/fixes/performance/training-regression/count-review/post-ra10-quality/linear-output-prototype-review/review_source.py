"""Independent source/ownership pin for one external scratch prototype; stdlib only."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
OWNER=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra10-quality/linear-output-mean-prototype'
BASE=ROOT/'integration/review/training-regression/post-ra9-quality/mean-category-production/run_mechanics_v2.py'
PACKAGE=ROOT/'pkg-CB64-RA10'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write_new(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')


def node(path,name):
    matches=[n for n in ast.walk(ast.parse(Path(path).read_text())) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(matches)==1,(str(path),name)
    return matches[0]


def dump(value):return ast.dump(value,include_attributes=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-preseal-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert args.output.resolve().parent==HERE and not args.output.exists()
    assert sha(OWNER/'SOURCE-FROZEN.json')==args.owner_preseal_sha256
    seal=read(OWNER/'SOURCE-FROZEN.json')
    guards=dict(seal['source_and_input_sha256'])
    for path,digest in guards.items():assert sha(path)==digest,path
    ready=read(ROOT/'quality/ra10/READY.json')
    assert ready['package_sha256']=='c8f8b343bded25ce8153bdd74ab9f722b369c0e087aee8a44d3c6e816baef2e4'
    assert len(ready['package_source_sha256'])==30
    for relative,digest in ready['package_source_sha256'].items():
        path=PACKAGE/'particlegan'/relative
        assert sha(path)==digest
        guards[str(path)]=digest
    for filename in ('linear_mean.py','run_prototype.py','raw_moments.py'):
        path=OWNER/filename
        assert str(path) in guards
        compile(path.read_text(),str(path),'exec')
    driver=OWNER/'run_prototype.py';module=OWNER/'linear_mean.py'
    assert sha(BASE)=='6f9fc07a8ceafbc9f60e086de32ea499e131e4e8d2bbc39da92e569c69e538b6'
    guards[str(BASE)]=sha(BASE)
    # Preserve the corrected constructor and helper copy/hash laws exactly.
    unchanged={}
    for name in ('construct','cpu','to_device','state_hash'):
        identical=dump(node(driver,name))==dump(node(BASE,name))
        assert identical,'corrected v2 raw-fidelity block changed: '+name
        unchanged[name]=hashlib.sha256(dump(node(driver,name)).encode()).hexdigest()
    # The paired candidate allocation is the old bounded algorithm with only
    # the objective input switched from critic metric to projected output.
    proposal=node(module,'propose_pairs')
    class RestoreMetric(ast.NodeTransformer):
        def visit_Attribute(self,value):
            self.generic_visit(value)
            if value.attr=='moment_metric':value.attr='metric'
            return value
    restored=RestoreMetric().visit(proposal)
    assert dump(restored)==dump(node(PACKAGE/'particlegan/mean_transport.py','propose_pairs'))
    text=driver.read_text()
    assert 'SOURCE-FROZEN-v2.json' not in text
    assert "schema='scratch_linear_output_v1'" in module.read_text()
    assert "purpose='nonresumable scratch reaction tensors, not backend9 state'" in text
    assert 'no_resumable_newlaw_checkpoint=True' in text
    assert 'check_state(after' not in text  # Altered-law after state is not a backend9 resume.
    assert 'packet.fast_output_metric' in text and 'packet.ema_output_metric' in text
    assert 'projected_outputs' in text and 'actual_packet_raw_outputs_exact' in text
    assert 'learned_cache_packet_features_exact' in text
    # The executable witness module is external; the immutable production
    # callbacks are restored in finally rather than any package file edit.
    assert 'fc.prepare_mean_witness,fc.run_mean_phase=baseline_prepare,baseline_phase' in text
    assert "source_preseal_sha256=sha(HERE/'SOURCE-FROZEN.json')" in text
    capture=node(module,'capture_outputs_features')
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model'
               for n in ast.walk(capture))==1
    fit=node(module,'fit_projection')
    calls=[n for n in ast.walk(fit) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in
                   ('randn','rand','manual_seed','set_state','normal','multinomial') for n in calls)
    step=node(PACKAGE/'particlegan/training.py','_step')
    hook=next(n for n in step.body if isinstance(n,ast.If) and
        ast.unparse(n.test)=='self.birth_death is not None' and
        any(isinstance(c,ast.Assign) and any(ast.unparse(t)=='event' for t in c.targets) for c in n.body))
    hook_digest=hashlib.sha256(dump(hook).encode()).hexdigest()
    proof=dict(status='PASS',scope='Source-only API/plumbing/raw-fixture ownership qualification',
        utc=datetime.now(timezone.utc).isoformat(),owner_preseal_sha256=args.owner_preseal_sha256,
        source_and_input_sha256=guards,all30_production_modules_unchanged=True,
        corrected_v2_blocks_AST_exact=unchanged,unchanged_trainer_caller_AST_sha256=hook_digest,
        allocation_inverse_restores_original_AST=True,
        reviewed_contract=dict(
            fresh_constructor_restores_G_D_EMA_tables_FIFO_row_moments_history_bandwidth_noise=True,
            old_trainer_backend_state_not_loaded_or_relabelled=True,
            deterministic_even_only_axes_no_action_stream_draw=True,
            moment_rank_separate_from_critic_chart_rank=True,
            raw_output_once_per_chunk_feeds_support_and_output_projection=True,
            projected_metric_used_for_ranking_refresh_preview_and_objective=True,
            same_group_denominators_refreshed_after_prefix_fixed_directions_retained=True,
            one_paired_jitter_draw_exact_prepared_commit_and_output_payload_digest=True,
            prefix_budget_sources_children_seeds_reserved_unique_disjoint=True,
            existing_postcommit_capture_checks_projected_payload_and_feature_cache=True,
            complete_copy_novel_mean_isolation_union_goes_to_original_rebase_and_own_evidence_reset=True,
            fresh_participation_limit_explicit=True,
            dedicated_streams_model_state_and_original_inputs_checked_neutral=True,
            new_law_export_marked_nonresumable_not_backend9_resume=True),
        Torch_imported=False,PT_objects_loaded=0,forwards=0,chart_fits=0,new_actions=0,
        runtime_API_or_quality_claim=False,execution_requires='root GO after both source reviews',
        limits=['source evidence only; one authorized owner runtime must establish actual assertions',
                'critic-conditioned assignment and clipping still remain',
                'fresh zero evidence/participation is not inherited historical evidence'])
    args.output.mkdir()
    write_new(args.output/'receipt.json',proof)
    local={str(HERE/'review_source.py'):sha(HERE/'review_source.py'),
           str(args.output/'receipt.json'):sha(args.output/'receipt.json'),
           str(OWNER/'SOURCE-FROZEN.json'):sha(OWNER/'SOURCE-FROZEN.json')}
    for path,digest in guards.items():assert sha(path)==digest,path
    write_new(args.output/'FROZEN.json',dict(status='PASS',scope='CLOSED_PRE_EXECUTION_SOURCE_REVIEW',
        utc=datetime.now(timezone.utc).isoformat(),receipt_sha256=sha(args.output/'receipt.json'),
        source_and_input_sha256=guards,local_sha256=local,numerical_execution=False))
    print('SOURCE_PASS',sha(args.output/'receipt.json'),sha(args.output/'FROZEN.json'),len(guards),flush=True)


if __name__=='__main__':main()
