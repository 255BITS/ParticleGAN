"""Declare the matched integration study; never launches training.

Run after all host/core commits have been integrated. Commit declarations and
source before run.py freezes requests. Historical cards stay reproducible from
the recorded develop commit; this revision changes explicit hook contracts only.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.planning import resolve_idea

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next')
BASE = '5737ade47dca89b04d338ada20667781d1f2b5df'
PREFIX = 'bcap-develop-integration'
WINNER = 'bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36'


def original(relative):
    return json.loads(subprocess.check_output(['git', 'show', f'{BASE}:{relative}'], cwd=ROOT))


CONTRACTS = {
    'two_pole': ('direct particle coordinates', 'fixed original two-pole target panel', 'original adversarial'),
    'unused_token_hold': ('original trained-concept output panel', 'repeated trained concept only; held-out token excluded', 'original adversarial and feature matching; existing hold'),
    'ae_gan_hold': ('generated decoder outputs only', 'original current data panel', 'original generated adversarial and cover; reconstruction'),
    'trajectory': ('original generated trajectory output panel', 'original current target panel; slow_arc excluded from distance', 'original adversarial; no paired MSE added'),
    'residual_student': ('original generated residual output panel', 'original current target panel; context excluded from distance', 'original adversarial; existing masked paired error'),
    'unipolar': ('original first-row outputs at two training scales', 'original scale-zero and scale-one targets; scales excluded', 'original adversarial'),
    'cover_leftover': ('original generated positive and negative conditional panels', 'original current real panels; labels excluded', 'original adversarial; existing paired cover'),
    'mid_scale_identity': ('original generated output panel at four training scales', 'original four target panels; scales excluded', 'original adversarial; existing paired cover'),
    'five_word_joint': ('generated categorical probabilities flattened per word', 'original current real word minibatch; latent coordinates excluded', 'original joint adversarial loss for G, encoder and prior'),
}


def main():
    view = original('configs/forge/views/discriminator_stability.json')
    selected = [a['task'] for a in view['assignments'] if a['qualification_tier'] <= 2]
    originals, changes = {}, []
    # The retained acquisition card still serves current software callers. Its
    # optional public fixture changed with the smoke/hold cards; rebind its
    # inactive-compatible source explicitly without adding a transport contract.
    for name in [*selected, 'five_word_joint_acquisition']:
        relative = f'configs/forge/tasks/{name}.json'
        old = original(relative)
        task = deepcopy(old)
        originals[name] = old
        host = task['execution'].get('host')
        if (name in selected and task['execution'].get('execution_path') == 'public_components'
                and host in CONTRACTS):
            space, target, protected = CONTRACTS[host]
            task['task_cohort'] = 'component_output_transport_v1'
            task['execution']['transport_consumer'] = 'output_marginal_v1'
            task['execution']['transport_contract'] = dict(version=1, sample_space=space,
                target_law=target, conditioning_in_distance=False, new_paired_supervision=False,
                new_forward_passes=False, new_random_draws=False,
                gradient_ownership='original generated branch parameters and learned prior; encoder receives only original word joint objective',
                protected_objectives=protected, inactive='original losses, updates, clocks and RNG; no hook requirement')
        sources = task['evaluation'].get('sources', {})
        replaced = {}
        for path, digest in list(sources.items()):
            if file_hash(ROOT/path) != digest:
                replaced[path] = digest
                sources[path] = file_hash(ROOT/path)
        if 'transport_consumer' in task['execution']:
            for path in ('particlegan/conditional_transport.py', 'particlegan/kinetic_transport.py',
                         'experiments/forge/behavior_adapters.py'):
                sources[path] = file_hash(ROOT/path)
            if host == 'five_word_joint':
                for path in ('experiments/forge/word_adapter.py', 'experiments/forge/word_tasks.py'):
                    sources[path] = file_hash(ROOT/path)
        if task != old:
            task['evaluation']['evaluator_revision'] = dict(id='opt-in-component-objective-hooks-v1',
                change='Explicit optional transport/protected-objective hooks and current source binding. Original question, architecture, data/prior/sampling laws, update budget, cadence and all numerical gates retained. Disabled mechanisms preserve original behavior.',
                original_task_commit=BASE, original_task_path=relative,
                previous_source_sha256=replaced, previous_revision=old['evaluation'].get('evaluator_revision'))
            (ROOT/relative).write_text(json.dumps(task, indent=2)+'\n')
            changes.append(name)
    atomic_json(OUT/'original-task-contracts.json', dict(schema_version=1, source_commit=BASE,
        qualification_input=False, tasks=originals))
    baseline = original(f'configs/forge/configurations/{WINNER}.json')
    for key in ('configuration_id', 'resolved_configuration_recipe', 'trainer_family'):
        baseline.pop(key, None)
    plans = {}
    for role in ('winner', 'combined'):
        card = deepcopy(baseline)
        card.update(id=f'{PREFIX}-{role}-v1', parent=WINNER,
            guide='reports/forge/bcap-develop-integration/README.md', mechanism_class='structural',
            changed_factors=['Matched develop host integration; original winner control' if role == 'winner' else
                'constraint_geometry_mode=direction_blend; kinetic_transport_weight=1; kinetic_transport_local_weight=1; kinetic_transport_projections=32'],
            mechanism_rationale='Reuse PR373 measured direction-only common descent and exact local-v2 on explicit opt-in hosts; no task-specific tuning.')
        if role == 'combined':
            card['recipe_overrides'].update(constraint_geometry_mode='direction_blend',
                kinetic_transport_weight=1., kinetic_transport_local_weight=1., kinetic_transport_projections=32)
        atomic_json(ROOT/f'configs/forge/ideas/{card["id"]}.json', card)
        study = dict(schema_version=1, id=f'{PREFIX}-{role}-study-v1', status='ready',
            candidate=card['id'], control=dict(candidate_id=f'{PREFIX}-{"combined" if role == "winner" else "winner"}-v1', task_map={}),
            hypothesis='The measured direction-only/local-v2 union can preserve the BCAP winner Tier1 passes and repair conditional identity and rare/broad density under one global recipe across all six Tier1 and21 Tier2 tasks.',
            competing_explanation='Marginal matching may conflict with conditional goals or degenerate targets, and first-order protected descent does not certify finite density or Gaussian retention.',
            scope=dict(view='discriminator_stability', through_tier=2, execution_backend='cuda', cuda_model='NVIDIA RTX A6000'),
            campaign=dict(id=f'{PREFIX}-v1', budget_seconds=90000, candidate_budget_seconds=45000),
            max_rounds=1,
            prior_evidence=[dict(path='reports/forge/bcap-develop-integration/prior-evidence.json', selector=[],
                identity=dict(research_publication_commit='fed122eed540a100a83d0b2614895f35d6c9e5c9'), use='motivation_only')],
            prediction=dict(task_id='vector_unequal_mass', metric='component_covariance_error', op='<=', threshold=.85, phase='final'),
            falsifier=dict(task_id='vector_unequal_mass', metric='component_covariance_error', op='>', threshold=.85, phase='final'),
            terminal_rules=dict(falsified='stop_revision', incomplete='request_missing_evidence',
                inconclusive='stop_and_readout', prediction_observed='review_saved_diagnostics'))
        atomic_json(ROOT/f'configs/forge/studies/{study["id"]}.json', study)
    for role in ('winner', 'combined'):
        request = resolve_idea(ROOT, f'{PREFIX}-{role}-v1', study=f'{PREFIX}-{role}-study-v1',
                               queue_root=ARCHIVE/'queue', freeze_source=False)
        plans[role] = dict(admission=request['study_review']['status'], blockers=request['preflight_blockers'],
            tasks={name: task['preflight_blockers'] for name, task in request['tasks'].items()},
            source_digest=request['source']['digest'], jobs=len(request['jobs']))
    atomic_json(OUT/'preregistration.json', dict(schema_version=1, qualification_input=False,
        source=BASE, original_winner=WINNER, arms=['winner', 'combined'], updated_tasks=changes,
        matched_conditions=['architecture', 'data law and actual batches', 'prior', 'sampling', 'initializer', 'budget', 'cadence', 'full numerical gates'],
        seed=0, global_config_per_arm=True, required_counts=dict(tier1=6,tier2=21),
        main_full_reservation_seconds=86040, main_campaign_ceiling_seconds=90000,
        software_allowance_seconds=3600,
        deeper_diagnostics='Only if ordinary Tier1 fails, preregister a separate admitted research diagnostic comparison for remaining Tier2 tasks. Own passing checkpoint producers remain mandatory. Diagnostic evidence grants no qualification.',
        stop='Complete measured revision and publish all gates, blockers, costs and compatibility. No seed experiments, tuning, merge or default promotion.', plans=plans))
    print(json.dumps(plans, indent=2))


if __name__ == '__main__':
    main()
