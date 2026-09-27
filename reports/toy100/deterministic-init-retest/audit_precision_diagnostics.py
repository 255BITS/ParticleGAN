"""Validate recorded rates, noise and game-field workload for three screen survivors."""
import ast
import json
from pathlib import Path

from audit_public3_runtime import checkpoint, read, sha

ROOT = Path(__file__).resolve().parent


def main():
    results = []
    for name in ('api-rp12', 'api-rp14', 'api-rp15'):
        audit = read(ROOT / f'{name}-runtime-audit.json')
        assert audit['status'] == audit['result'] == 'PASS'
        source = Path(audit['source'])
        initial, _ = checkpoint(source / 'initial-state.pt')
        final, _ = checkpoint(source / 'final-state.pt')
        recipe = final['trainer']['recipe']
        assert recipe['total_steps'] is None and recipe['noise_policy'] == 'constant'
        assert recipe['continuous_precision'] == 'rp5'
        package = ROOT / 'port-source' / name / 'package'
        game_source = package / 'particlegan/game_update.py'
        declaration = read(source / 'declaration.json')
        assert sha(game_source.read_bytes()) == declaration['package_sha256']['particlegan/game_update.py']
        three = next(ast.literal_eval(n.value) for n in ast.parse(game_source.read_text()).body
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'THREE_FIELD_POLICIES' for t in n.targets))
        fields = 3 if recipe['game_update'] in three else 2
        before = initial['trainer']['precision']['state']
        after = final['trainer']['precision']['state']
        assert before['open'] and before['updates'] == 0
        assert after['updates'] == 1200
        rows = [json.loads(line) for line in (source / 'learning-rates.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(1, 1201))
        rate_states, changes, previous = [], [], None
        for row in rows:
            rates = [[g['lr'] for g in role] for role in row['applied_group_rates']]
            choices = {}
            for label, net, prior in (('open', 1., 1.), ('reduced', .01, .05)):
                choices[label] = [[recipe['lr'] * net, recipe['lr'] * recipe['prior_lr_mult'] * prior],
                                  [recipe['lr'] * recipe['d_lr_mult'] * net]]
            matches = [label for label, expected in choices.items() if expected == rates]
            assert len(matches) == 1
            current = matches[0]
            rate_states.append(current)
            if current != previous:
                changes.append(dict(first_update=row['step'], state=current, rates=rates))
            previous = current
            assert row['input_noise'] == recipe['input_noise_std'] == 0.
            assert row['output_noise'] == recipe['output_noise_std'] == .029
            stats = row['game_stats']
            assert stats['policy'] == recipe['game_update']
            assert stats['accepted_updates'] == row['step'] and stats['field_evaluations'] == fields
        closes = sum(c['state'] == 'reduced' for c in changes)
        reopens = sum(c['state'] == 'open' for c in changes[1:])
        # Rates for update t precede the observation after that update.
        closes += after['event'] == 'close'
        reopens += after['event'] == 'reopen'
        assert after['closings'] == closes and after['openings'] == 1 + reopens
        results.append(dict(candidate=declaration['candidate'], transitions=changes,
            final_precision=after, nominal_game_fields=1200 * fields,
            rate_rows_verified=1200, source=source.as_posix(),
            rates_sha256=sha((source / 'learning-rates.jsonl').read_bytes()),
            final_state_sha256=sha((source / 'final-state.pt').read_bytes()),
            source_game_update_sha256=sha(game_source.read_bytes())))
    report = dict(status='PASS_RECORDED_DIAGNOSTICS', candidates=results,
        limits=['Checks nominal applied group rates, constant noise, source-bound game-field counters and final controller transition counts; no learner or floating-point controller replay.',
                'Tiny harness does not emit every precision state, so raw gap/activity smoothing is not independently reconstructed.',
                'Field count excludes auxiliary controller/scoring work and does not establish runtime superiority.',
                'No target change occurs here. A reduced rate with retained quality does not prove autonomous recovery.',
                'RP14 inherited force-cosine diagnostic limitations remain; this audit does not validate that metric.'])
    (ROOT / 'precision-diagnostics-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([dict(candidate=r['candidate'], transitions=[dict(first_update=c['first_update'], state=c['state']) for c in r['transitions']], fields=r['nominal_game_fields']) for r in results]))


if __name__ == '__main__':
    main()
