"""Compare five complete, source-bound publications without executing training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TRACKS = ('optimistic', 'anisotropic', 'sinkhorn', 'secant', 'confidence')
VERDICTS = {'PASS', 'FAIL'}


def read(path):
    return json.loads(path.read_text())


def publication_summary(path, audit_path):
    """Separate individual certified grades from verified matched consumption."""
    result = read(path)
    audit = read(audit_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['scope'] == 'phase3_paired_research_diagnostic'
    assert result['qualification_input'] is False and result['protocol_seed'] == 0
    assert result['original_placement_counts'] == dict(tier1=6, tier2=10)
    assert audit['status'] == 'PASS' and audit['task_results_sha256'] == digest
    assert all(audit[key] == result[key] for key in ('source_commit', 'source_digest'))
    cells = {(cell['role'], cell['task_id']): cell for cell in result['task_cells']}
    assert len(cells) == 32
    tasks = {cell['task_id'] for cell in cells.values()}
    groups = {item['task_id']: item for item in audit['saved_state_comparisons']
              if item['saved_variant'] == 'final'}
    producers = {(item['role'], item['task_id']): item['prefix_steps']
                 for item in audit['own_checkpoint_producers']}
    categories = ('repaired', 'retained_pass', 'regressed', 'persistent_failure', 'unresolved')
    certified = {name: [] for name in categories}
    verified = {name: [] for name in categories}
    pairs, unverified = [], []
    counts = {role: {str(tier): dict(PASS=0, FAIL=0, unresolved=0) for tier in (1, 2)}
              for role in ('baseline', 'candidate')}
    for task in sorted(tasks):
        baseline, candidate = (cells[(role, task)] for role in ('baseline', 'candidate'))
        assert baseline['tier'] == candidate['tier']
        tier = baseline['tier']
        for role, cell in (('baseline', baseline), ('candidate', candidate)):
            verdict = cell['gate_status']
            counts[role][str(tier)][verdict if verdict in VERDICTS else 'unresolved'] += 1
        statuses = (baseline['gate_status'], candidate['gate_status'])
        category = {('FAIL', 'PASS'): 'repaired', ('PASS', 'PASS'): 'retained_pass',
                    ('PASS', 'FAIL'): 'regressed', ('FAIL', 'FAIL'): 'persistent_failure'}.get(statuses, 'unresolved')
        item = dict(task_id=task, original_tier=tier, baseline=statuses[0], candidate=statuses[1])
        certified[category].append(item)
        group = groups.get(task, {})
        matched = (category != 'unresolved' and all(group.get(key) is True for key in
                   ('consumed_non_eval_streams_and_batches_equal', 'initialization_and_prior_equal',
                    'named_training_bindings_equal')))
        pair = dict(item, consumed_stream_pair_status='VERIFIED' if matched else 'UNVERIFIED',
                    completed_steps=group.get('completed_steps', {}),
                    own_prefix_steps={role: producers[(role, task)] for role in ('baseline', 'candidate')
                                      if (role, task) in producers},
                    reason=group.get('consumption_comparison', 'No complete paired saved-state group.'))
        pairs.append(pair)
        if matched:
            verified[category].append(item)
        else:
            unverified.append(pair)
    assert all(sum(counts[role][str(tier)].values()) == total
               for role in counts for tier, total in ((1, 6), (2, 10)))
    repaired = [item['task_id'] for item in verified['repaired'] if item['original_tier'] == 2]
    certified_repairs = [item['task_id'] for item in certified['repaired'] if item['original_tier'] == 2]
    regressed = [item['task_id'] for item in certified['regressed'] if item['original_tier'] == 2]
    gate_eligible = (counts['candidate']['1']['PASS'] == 6 and bool(certified_repairs)
                     and not regressed and not certified['unresolved'])
    eligible = gate_eligible and bool(repaired) and not unverified
    paid = sum(item['selected_paid_seconds'] for item in result['accounting'])
    assert paid <= result['paid_ceiling_seconds'] == 48000
    assert abs(audit['paid_seconds'] - paid) < 1e-7
    assert audit['actual_training_gifs'] == result['actual_training_gifs']
    return dict(publication_path=str(path.parent), source_commit=result['source_commit'],
        source_digest=result['source_digest'], results_sha256=digest,
        independent_audit_path=str(audit_path),
        independent_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        comparison_basis='Certified individual gate statuses; matched effects require verified consumed streams.',
        outcomes=counts, certified_gate_changes=certified, verified_consumption_changes=verified,
        consumed_stream_pairs=pairs, unverified_consumption_pairs=unverified,
        certified_tier2_repairs=certified_repairs, tier2_repairs=repaired, tier2_regressions=regressed,
        gate_selection_eligible=gate_eligible, eligible_research_baseline=eligible,
        paid_seconds_including_predecessors=paid,
        execution_retries=sum(item['execution_retries'] for item in result['accounting']),
        actual_training_gifs=result['actual_training_gifs'])


def compare(root, audit_root=None):
    audit_root = audit_root or Path(__file__).resolve().parent
    summary = dict(schema_version=1, scope='five_paired_research_diagnostics',
                   qualification_input=False, ordinary_tier2_questions_unmeasured=11,
                   selection_rule='Six Tier1 passes, at least one paired complete Tier2 repair, '
                                  'no regression of a baseline Tier2 pass, and no unresolved comparisons.',
                   tracks={}, eligible_research_baselines=[])
    for track in TRACKS:
        report = Path(str(root) + '-bcap-moonshot-' + track) / 'reports/forge' / ('bcap-moonshot-' + track)
        publications = [path for path in (report, report / 'publication')
                        if (path / 'phase3-results.json').exists()]
        assert len(publications) == 1, f'{track} needs one unambiguous final publication'
        item = publication_summary(publications[0] / 'phase3-results.json',
                                   audit_root / f'phase3-{track}-independent-audit.json')
        item['pr'] = 'https://github.com/255BITS/ParticleGAN/pull/' + dict(
            optimistic='378', anisotropic='380', sinkhorn='379', secant='381', confidence='382')[track]
        summary['tracks'][track] = item
        if item['eligible_research_baseline']:
            summary['eligible_research_baselines'].append(track)
    summary['paid_seconds'] = sum(item['paid_seconds_including_predecessors'] for item in summary['tracks'].values())
    summary['interpretation'] = ('Paired full-gate comparisons remain separated by frozen source. '
        'No pooled runtime ranking, endpoint rescue, ordinary qualification, automatic merge or default adoption. '
        'Confidence retains its separately recorded software allowance violation.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-repository', type=Path, default=Path('/home/martyn/dev/ParticleGAN'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, default=Path(__file__).resolve().parent)
    options = parser.parse_args()
    result = compare(options.main_repository.resolve(), options.audit_root.resolve())
    options.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(eligible=result['eligible_research_baselines'], paid_seconds=result['paid_seconds'])))


if __name__ == '__main__':
    main()
