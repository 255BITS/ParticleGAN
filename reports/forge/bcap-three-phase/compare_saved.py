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


def compare(root):
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
        publication = publications[0]
        path = publication / 'phase3-results.json'
        result = read(path)
        assert result['scope'] == 'phase3_paired_research_diagnostic'
        assert result['qualification_input'] is False and result['protocol_seed'] == 0
        assert result['original_placement_counts'] == dict(tier1=6, tier2=10)
        cells = {(cell['role'], cell['task_id']): cell for cell in result['task_cells']}
        assert len(cells) == 32
        tasks = {cell['task_id'] for cell in cells.values()}
        changes = {name: [] for name in ('repaired', 'retained_pass', 'regressed', 'persistent_failure', 'unresolved')}
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
            changes[category].append(dict(task_id=task, original_tier=tier,
                                          baseline=statuses[0], candidate=statuses[1]))
        assert all(sum(counts[role][str(tier)].values()) == total
                   for role in counts for tier, total in ((1, 6), (2, 10)))
        repaired = [item['task_id'] for item in changes['repaired'] if item['original_tier'] == 2]
        regressed = [item['task_id'] for item in changes['regressed'] if item['original_tier'] == 2]
        eligible = (counts['candidate']['1']['PASS'] == 6 and bool(repaired)
                    and not regressed and not changes['unresolved'])
        paid = sum(item['selected_paid_seconds'] for item in result['accounting'])
        assert paid <= result['paid_ceiling_seconds'] == 48000
        summary['tracks'][track] = dict(publication_path=str(publication), source_commit=result['source_commit'],
            source_digest=result['source_digest'], results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            outcomes=counts, paired_changes=changes, tier2_repairs=repaired, tier2_regressions=regressed,
            eligible_research_baseline=eligible, paid_seconds_including_predecessors=paid,
            execution_retries=sum(item['execution_retries'] for item in result['accounting']),
            actual_training_gifs=result['actual_training_gifs'],
            pr='https://github.com/255BITS/ParticleGAN/pull/' + dict(
                optimistic='378', anisotropic='380', sinkhorn='379', secant='381', confidence='382')[track])
        if eligible:
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
    options = parser.parse_args()
    result = compare(options.main_repository.resolve())
    options.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(eligible=result['eligible_research_baselines'], paid_seconds=result['paid_seconds'])))


if __name__ == '__main__':
    main()
