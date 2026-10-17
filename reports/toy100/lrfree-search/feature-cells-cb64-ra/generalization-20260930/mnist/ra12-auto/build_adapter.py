"""Prepare an auditable API adapter from the frozen original fixture runners."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LANE = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/validation-cb64-ra11/learned')
NAME = 'RA12-auto'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, before, after, edits):
    assert text.count(before) == 1, before
    edits.append(dict(before=before, after=after))
    return text.replace(before, after)


def main():
    assert not (ROOT / 'SOURCE-FREEZE.json').exists(), 'adapter already frozen'
    edits = []
    source = (LANE / 'run_training.py').read_text()
    source = replace_once(source,
        'from models_metrics import networks, Evaluator, encode, draw_samples, toy_metrics, model_hash, tensor_hash, manifold_precision_recall',
        'from models_metrics import networks, Evaluator, encode, model_hash, tensor_hash, manifold_precision_recall\n'
        'from adapter import draw_samples, toy_metrics, initialize_fixture_models, selection_diagnostics\n'
        'from contracts import validate_checkpoint_state, validate_initial_streams, verify_fixture_sources, toy_gate', edits)
    source = replace_once(source, "            recipe=Recipe(**config)",
        "            require('initialization' not in config, 'current public Recipe owns no initialization')\n"
        "            fixture_sources=verify_fixture_sources()\n"
        "            recipe=Recipe(**config)", edits)
    source = replace_once(source,
        '            prior=ParticlePrior(1024,128,generator=torch.Generator().manual_seed(SEED+1))',
        '            prior=ParticlePrior(1024,128,generator=torch.Generator().manual_seed(SEED+1))\n'
        '            initialize_fixture_models(G,D)', edits)
    source = replace_once(source,
        "            require(initial==previous, f'initial G/D/prior differ from previous GPU E22: {initial}')",
        "            require(initial==previous, f'initial G/D/prior differ from previous GPU E22: {initial}')\n"
        "            initial_streams=validate_initial_streams(torch,trainer)", edits)
    source = replace_once(source,
        '                         previous_gpu_initialization_verified=True,device=DEVICE,',
        '                         previous_gpu_initialization_verified=True,device=DEVICE,\n'
        '                         fixture_sources=fixture_sources,initial_streams=initial_streams,\n'
        "                         primary_sampling={'output_noise':True,'scorer':'unchanged original'},", edits)
    source = replace_once(source,
        '    return out\n\n\ndef main():',
        "    out['backend_selection']=selection_diagnostics(trainer)\n"
        "    if trainer.policy.surprise is not None:out['surprise']=trainer.policy.surprise.diagnostics()\n"
        '    return out\n\n\ndef main():', edits)
    source = replace_once(source,
        '                state=trainer.state_dict()\n                record[\'rng_buffer_placement\']=rng_cpu_buffers(torch,state)',
        '                state=trainer.state_dict()\n'
        "                record['backend_selection']=validate_checkpoint_state(state,trainer.policy.roles,args.problem,recipe)\n"
        "                require(record['backend_selection']==record['diagnostics']['backend_selection'],'record/checkpoint selection disagrees')\n"
        "                record['rng_buffer_placement']=rng_cpu_buffers(torch,state)", edits)
    source = replace_once(source,
        "            write_json(outdir/'result.json',result)",
        "            result['original_quality_gate']=('PASS' if toy_gate(curves[-1]['metrics']) else 'FAIL') if args.problem=='toy' else None\n"
        "            write_json(outdir/'result.json',result)", edits)
    ast.parse(source)
    (ROOT / 'run_training.py').write_text(source)
    replay_edits = []
    replay = (LANE / 'replay.py').read_text()
    replay = replace_once(replay, 'from models_metrics import networks',
        'from models_metrics import networks\nfrom contracts import validate_checkpoint_state\n'
        'from adapter import draw_samples', replay_edits)
    placements = '''def tensor_placements(value):
    """Read all tensor locations, including CPU RNG/Adam step buffers."""
    rows=[]
    def visit(item,path):
        if isinstance(item,torch.Tensor):
            rows.append(dict(path=path,device=str(item.device),dtype=str(item.dtype),shape=list(item.shape)))
        elif isinstance(item,dict):
            for key,child in item.items():visit(child,path+'.'+str(key))
        elif isinstance(item,(list,tuple)):
            for index,child in enumerate(item):visit(child,path+'.'+str(index))
    visit(value,'trainer')
    return rows


'''
    replay = replace_once(replay, 'def digest(value):', placements + 'def digest(value):', replay_edits)
    replay = replace_once(replay, '            trainer.load_state_dict(saved)',
        "            restore_input=(saved if branch==0 else torch.load(path,map_location='cpu',weights_only=False)['trainer'])\n"
        '            trainer.load_state_dict(restore_input)', replay_edits)
    replay = replace_once(replay, '            restored=trainer.state_dict()',
        '            restored=trainer.state_dict()\n'
        "            validate_checkpoint_state(restored,trainer.policy.roles,problem,trainer.recipe)", replay_edits)
    replay = replace_once(replay,
        "            restored_exact['whole_state']=digest(semantic_state(restored))==saved_semantic_sha",
        "            restored_semantic_sha=digest(semantic_state(restored))\n"
        "            restored_exact['whole_state']=restored_semantic_sha==saved_semantic_sha", replay_edits)
    replay = replace_once(replay,
        '                        restored_state_semantic_sha256=digest(semantic_state(restored)),',
        '                        restored_state_semantic_sha256=restored_semantic_sha,', replay_edits)
    replay = replace_once(replay,
        '            phase=f\'save_branch_{branch}\'\n            branch_path=outdir/f\'branch-{branch}.pt\'',
        "            phase=f'primary_sample_branch_{branch}'\n"
        "            sample_count=8192 if problem=='toy' else 4096\n"
        "            samples=draw_samples(trainer,sample_count,SEED+100).detach().cpu().clone()\n"
        "            samples_sha=digest(samples)\n"
        "            sample_state_unchanged=digest(semantic_state(trainer.state_dict()))==semantic_sha\n"
        "            phase=f'save_branch_{branch}'\n            branch_path=outdir/f'branch-{branch}.pt'", replay_edits)
    replay = replace_once(replay,
        '            torch.save(dict(trainer=state,loss_tensors=loss_bits,data_position=2*(START+COUNT)*128,',
        '            torch.save(dict(trainer=state,loss_tensors=loss_bits,primary_samples=samples,data_position=2*(START+COUNT)*128,', replay_edits)
    replay = replace_once(replay, '            output=dict(branch=branch,losses=losses,',
        "            output=dict(branch=branch,checkpoint_load_map_location='native' if branch==0 else 'cpu',\n"
        "                        restore_input_tensor_placement=tensor_placements(restore_input),\n"
        "                        restored_tensor_placement=tensor_placements(restored),\n"
        "                        primary_samples_sha256=samples_sha,primary_sample_count=sample_count,\n"
        "                        primary_sample_seed=SEED+100,primary_sampling_output_noise=True,\n"
        "                        sample_preserves_semantic_training_state=sample_state_unchanged,losses=losses,", replay_edits)
    replay = replace_once(replay,
        "        required=(all(semantic_same.values()) and losses_same and restoration_exact",
        "        samples_same=outputs[0]['primary_samples_sha256']==outputs[1]['primary_samples_sha256']\n"
        "        samples_preserve_state=all(o['sample_preserves_semantic_training_state'] for o in outputs)\n"
        "        required=(samples_same and samples_preserve_state and all(semantic_same.values()) and losses_same and restoration_exact", replay_edits)
    replay = replace_once(replay,
        "                    losses_bit_identical=losses_same,sections_bit_identical=same_sections,",
        "                    primary_sample_bytes_bit_identical=samples_same,primary_sampling_preserves_training_state=samples_preserve_state,\n"
        "                    losses_bit_identical=losses_same,sections_bit_identical=same_sections,", replay_edits)
    ast.parse(replay)
    (ROOT / 'replay.py').write_text(replay)
    common = (LANE / 'common.py').read_text()
    assert common.count('VARIANTS = ("CB64-RA11",)') == 1
    common = common.replace('VARIANTS = ("CB64-RA11",)', 'VARIANTS = ' + repr((NAME,)))
    (ROOT / 'common.py').write_text(common)
    (ROOT / 'logs').mkdir(exist_ok=True)
    receipt = dict(status='SOURCE_ADAPTER_PREPARED_NO_NUMERICAL_EXECUTION',
                   original_runner_sha256=sha(LANE / 'run_training.py'),
                   adapted_runner_sha256=sha(ROOT / 'run_training.py'), edits=edits,
                   original_replay_sha256=sha(LANE / 'replay.py'),
                   adapted_replay_sha256=sha(ROOT / 'replay.py'), replay_edits=replay_edits,
                   common_only_change='VARIANTS names', no_extra_begin_step=True,
                   training_budget_unchanged=True, scorer_changes=[])
    (ROOT / 'source-transform-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    from contracts import verify_fixture_sources
    checked = verify_fixture_sources()
    preparation = dict(status='PREPARED_WAITING_FOR_ROOT_PACKAGE_REVIEW_FREEZE',
                       source_contract=checked, shared_config=str(ROOT.parents[1] / 'configs/RA12-auto.json'),
                       proposed_commands=[['/tmp/pr38-default-env/bin/python','-u','-B',str(ROOT / 'launch_first.py'),
                                           '--problem',problem] for problem in ('toy','mnist')],
                       training_updates=0,cuda_contexts=0,root_authorized_gpu=False,
                       local_sources={path.name:sha(path) for path in sorted(ROOT.glob('*.py'))})
    (ROOT / 'ADAPTER-PREPARATION.json').write_text(json.dumps(preparation, indent=2) + '\n')
    print(json.dumps(dict(status=preparation['status'], source_contract=checked)))


if __name__ == '__main__':
    main()
