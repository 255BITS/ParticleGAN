"""Read-only receipt consistency audit. Does not train, grade or rewrite results."""
import argparse, hashlib, json, math, operator, sys
from collections import Counter
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def filehash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--queue-root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    # Reuse serialization identities only; never invoke grade_result or a scorer.
    from experiments.forge.contracts import stable_hash
    from experiments.forge.planning import candidate_revision_for, load_idea
    global digest
    digest = stable_hash
    queue = (args.queue_root or root / 'runs/forge').resolve()
    report = root / 'reports/forge/tier1-completion'
    definition = read(root / 'configs/forge/rounds/tier1-completion-v1.json')
    results = read(report / 'results.json')
    media = read(report / 'media.json')['items']
    state = read(queue / 'queue/state.json')
    roster = read(queue / 'tier1-completion-v1/roster.json')
    campaign = definition['id']
    errors, notes = [], []
    checks = 0
    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)
    frozen = {r['family']: r for r in definition['candidate_roster']}
    published = {r['family']: r for r in results['candidates']}
    submissions = {r['family']: r for r in roster}
    check(len(frozen) == len(published) == len(submissions) == 11, 'exactly 11 recipes required')
    check(set(frozen) == set(published) == set(submissions), 'family roster differs')
    check(sum(len(r['tasks']) for r in published.values()) == 77, '77 question cells required')
    media_by_key = {}
    for item in media:
        key = (item['family'], item['task_id'], item['attempt_id'])
        check(key not in media_by_key, 'duplicate media key ' + str(key))
        media_by_key[key] = item
        path = (root / item['gif']).resolve()
        check(path.is_relative_to(report.resolve()), 'media path escaped report')
        check(path.is_file() and filehash(path) == item['gif_sha256'], 'GIF hash differs ' + str(key))
        receipt = read(path.with_suffix('.json'))
        check(receipt == {k: v for k, v in item.items() if k not in {'family', 'attempt_id', 'gif'}}, 'media sidecar differs ' + str(key))
        for filename, sha in item['source_inputs'].items():
            check(Path(filename).is_file() and filehash(filename) == sha, 'saved media input differs ' + str(key))
        check(item['optimizer_updates_added'] == item['sampling_draws_added'] == 0, 'export added execution ' + str(key))
    sources, attempt_ids, blockers = set(), set(), []
    statuses = Counter()
    supported_policy = {'gaussian1d_acquisition', 'two_pole', 'ring16_acquisition', 'clockfree_audit'}
    blocked_policy = {'unused_token_hold', 'ae_gan_hold', 'five_word_joint_acquisition'}
    for family, pin in frozen.items():
        pub = published[family]
        entry = state['submissions'][submissions[family]['request_id']]
        req = entry['request']
        sources.add(req['source']['digest'])
        check(req['candidate']['id'] == pin['candidate_id'] == pub['candidate_id'], family + ': selected candidate differs')
        declaration = load_idea(root, pin['candidate_id'])
        check(digest(declaration) == pin['declaration_sha256'], family + ': selected declaration differs')
        for key, value in declaration.items():
            if key == 'prior':
                check(all(req['candidate']['prior'].get(k) == v for k, v in value.items()), family + ': declared prior differs')
            else:
                check(req['candidate'].get(key) == value, family + ': declared binding differs: ' + key)
        check(req['candidate_revision'] == candidate_revision_for(req['source']['digest'], req['candidate']), family + ': revision differs')
        check(digest(req['view']) == pin['view_fingerprint'], family + ': selected view differs')
        check(req['through_tier'] == 1, family + ': later tier authorized')
        check(req['execution_policy']['mode'] == 'complete_current_tier', family + ': execution mode differs')
        check(pub['source'] == req['source']['digest'] and pub['source_commit'] == req['source']['origin_commit'], family + ': published source differs')
        check(entry['status'] not in {'pending', 'running', 'queued'}, family + ': submission unfinished')
        tasks = {t['task_id']: t for t in pub['tasks']}
        check(set(tasks) == set(pin['task_ids']), family + ': question roster differs')
        assignments = {a['task']: a for a in req['view']['assignments']}
        for name, task in req['tasks'].items():
            if name in pin['task_definition_sha256']:
                shape = {k: v for k, v in task.items() if k not in {'field_ownership', 'preflight_blockers'}}
                check(digest(shape) == pin['task_definition_sha256'][name], family + '/' + name + ': frozen task differs')
        for job in req['jobs']:
            actual = state['jobs'][job['compatibility_key']]
            members = job.get('task_ids', [job['task_id']])
            eligible = all(assignments[n]['qualification_tier'] <= 1 for n in members)
            if not eligible:
                check(not actual['attempts'], family + ': later-tier job executed')
                continue
            check(len(actual['attempts']) <= 2, family + ': retry cap exceeded')
            if len(actual['attempts']) == 2:
                check(bool(actual.get('retry_of')) or bool(actual.get('result', {}).get('retry_of')), family + ': undocumented retry')
            for attempt in actual['attempts']:
                attempt_ids.add(attempt['attempt_id'])
        for name, row in tasks.items():
            label = family + '/' + name
            statuses[row['status']] += 1
            task = req['tasks'][name]
            preflight = task.get('preflight_blockers', [])
            if preflight:
                blockers.append((family, name))
                check(row['status'] == 'BLOCKED' and row['attempt_id'] is None, label + ': preflight blocker paid/measured')
                check(family in {'atlas', 'e22'} and task['policy_parent']['id'] in blocked_policy, label + ': unexpected explicit blocker')
                continue
            check(row['status'] != 'UNKNOWN', label + ': unfinished cell')
            attempt_id = row['attempt_id']
            check(attempt_id in attempt_ids, label + ': selected attempt not in queue')
            directory = root / 'reports/forge/attempts' / attempt_id
            durable, certificate = read(directory / 'result.json'), read(directory / 'evidence.json')
            envelope = read(directory / 'request.json')
            origin = envelope.get('request', envelope)
            check(certificate['result_hash'] == digest(durable) == row['canonical_result_hash'], label + ': result certificate differs')
            check(certificate['source'] == req['source'] == origin['source'], label + ': source certificate differs')
            check(certificate.get('runtime') == req.get('runtime'), label + ': runtime certificate differs')
            check(durable['candidate_revision'] == req['candidate_revision'], label + ': candidate receipt differs')
            local = Path(certificate['local_artifact_root'])
            check(read(local / 'result.json') == durable, label + ': local/durable result differs')
            measured = next(t for t in durable['task_results'] if t['task_id'] == name)
            check(measured['gate_status'] == row['status'] and measured.get('metrics', {}) == row['metrics'], label + ': published grade/metrics differ')
            if row['status'] not in {'PASS', 'FAIL'}:
                notes.append(label + ': ' + row['status'] + ' (no required numeric GIF)')
                continue
            grading = durable['raw']['grading']
            check(grading['raw_hash'] == digest(durable['raw']['result']) and grading['source_digest'] == req['source']['digest'], label + ': frozen evaluator certificate differs')
            grade = grading['grades'][name]
            check(grade.get('gate_status', grade.get('status')) == row['status'], label + ': independent grade differs')
            check(grade.get('metrics', {}) == measured.get('metrics', {}), label + ': evaluator metrics differ')
            evidence = measured['evidence']
            key = (family, name, attempt_id)
            check(key in media_by_key, label + ': numeric measurement lacks actual-training GIF')
            if key in media_by_key:
                check(media_by_key[key]['recorded_grade'] == row['status'], label + ': GIF grade differs')
                if task['adapter'] != 'clockfree_audit':
                    check(media_by_key[key]['observations_sha256'] == digest(evidence['observations']), label + ': GIF observations differ')
            if task['evaluation']['kind'] == 'transfer_sustained':
                observations = evidence['observations']
                expected = [math.ceil(i * task['execution']['steps'] / 24) for i in range(1, 25)]
                check([p['step'] for p in observations] == expected, label + ': observation schedule differs')
                threshold = task['evaluation']['thresholds']
                ops = {'>=': operator.ge, '<=': operator.le, '==': operator.eq}
                def passes(point):
                    return all(type(point.get(k)) in (int, float) and math.isfinite(point[k]) and ops[op](point[k], bound) for k, op, bound in threshold)
                suffix = 0
                for point in reversed(observations):
                    if not passes(point):
                        break
                    suffix += 1
                certified_evaluator = grade.get('evaluator_result')
                if not certified_evaluator:
                    guards = evidence.get('guards', {})
                    expected_guards = task['evaluation'].get('guards', {})
                    guard_failed = (expected_guards.get('finite_state') and guards.get('all_finite') is False) or any(guards.get('optimizer_updates', {}).get(role) == 0 for role in expected_guards.get('optimizer_roles', []))
                    check(row['status'] == 'FAIL' and guard_failed, label + ': missing numerical evaluator without certified guard failure')
                    notes.append(label + ': declared guard FAIL; no sustained-threshold verdict attributed')
                    continue
                convergence = certified_evaluator['convergence']
                check(convergence['passing_suffix'] == suffix and convergence['observations'] == 24 and convergence['complete'], label + ': certified terminal-suffix arithmetic differs')
                expected_pass = suffix >= task['evaluation']['minimum_stable_checks'] and passes(evidence['live'])
                check(expected_pass == (row['status'] == 'PASS'), label + ': certified threshold verdict inconsistent')
                check(grade['evaluator_result']['metrics'] and len(grade['evaluator_result']['metrics']) == len(threshold), label + ': certified bound receipts missing')
            elif task['evaluation']['kind'] == 'clockfree_parity':
                comparisons = evidence['comparisons']
                check(sorted(c['condition'] for c in comparisons) == sorted(task['evaluation']['conditions']), label + ': clock conditions differ')
                different = any(c['reference_sha256'] != c['changed_sha256'] for c in comparisons)
                dependencies = evidence['source_audit']['unexplained_clock_dependencies']
                check((not different and not dependencies) == (row['status'] == 'PASS'), label + ': certified clock verdict inconsistent')
            if family in {'atlas', 'e22'}:
                check(task['policy_parent']['id'] in supported_policy and task['task_cohort'] == 'tier1_policy_selected_cloud_v1', label + ': unsupported policy measurement/cohort')
    check(len(sources) == 1, 'multiple implementation cohorts')
    check(len(blockers) == 6, 'expected exactly six explicit policy blockers')
    charges = [c for c in state.get('charges', []) if c['owner']['campaign'] == campaign]
    check({c['attempt_id'] for c in charges} == attempt_ids, 'charged/executed attempt set differs')
    check(len(charges) == len({c['attempt_id'] for c in charges}), 'duplicate charges')
    charged = sum(c['seconds'] for c in charges)
    accounting = state['campaigns'][campaign]
    check(math.isclose(charged, accounting['spent_seconds'], rel_tol=1e-10, abs_tol=1e-6), 'total charges differ')
    check(results['accounting'] == accounting, 'published accounting differs')
    check(abs(accounting['reserved_seconds']) < 1e-6, 'paid reservations still active')
    check(charged <= definition['campaign']['budget_seconds'], 'campaign ceiling exceeded')
    per_candidate = Counter()
    for c in charges:
        per_candidate[state['submissions'][c['owner']['request']]['request']['candidate']['id']] += c['seconds']
    check(all(v <= definition['campaign']['candidate_budget_seconds'] for v in per_candidate.values()), 'candidate ceiling exceeded')
    print(json.dumps({'status': 'FAIL' if errors else 'PASS', 'checks': checks, 'cells': sum(statuses.values()), 'statuses': statuses, 'explicit_blockers': blockers, 'source_digests': sorted(sources), 'attempts': len(attempt_ids), 'charged_seconds': charged, 'errors': errors, 'notes': notes}, sort_keys=True, indent=2))
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
