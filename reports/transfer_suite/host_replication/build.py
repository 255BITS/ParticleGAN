"""Rebuild the second-host replication report from archived episodes; no training."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from benchmarks.transfer_suite.protocol import test_verdict
from .run import SUITE, untimed

ROOT = Path(__file__).resolve().parent
CLI_RUNS = [('default', 'CPU wheel', 'none'), ('aten_avx2', 'CPU wheel', 'ATen AVX2'),
            ('all_avx2', 'CPU wheel', 'ATen+MKL+oneDNN AVX2'), ('cu126_default', 'cu126 wheel (archive build)', 'none'),
            ('cu126_aten_avx2', 'cu126 wheel (archive build)', 'ATen AVX2'),
            ('cu126_all_avx2', 'cu126 wheel (archive build)', 'ATen+MKL+oneDNN AVX2')]
REFERENCE = SUITE / 'rare_focus/cli_replay/reference.json.gz'


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def observed(result, spec):
    verdict = test_verdict(spec, result)
    return dict(status=verdict['status'], passing_suffix=verdict['convergence']['passing_suffix'],
                confirmed_step=verdict['convergence']['confirmed_step'],
                final_failing=[m['metric'] for m in verdict['metrics'] if m['status'] != 'PASS'])


def build():
    plan = read(ROOT / 'plan.json.gz')
    protocol = read(ROOT / 'protocol.json.gz')
    index = read(ROOT / 'index.json.gz')['records']
    assert [r['job'] for r in index] == plan['jobs'] and len(index) == 46
    cases = defaultdict(list)
    for record in index:
        value = read(ROOT / record['artifact'])
        source = SUITE / record['job']['artifact']
        assert value['source_artifact_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        replay = observed(value['result'], value['spec'])
        assert replay == value['comparison']['replay'] and value['verdict'] == test_verdict(value['spec'], value['result'])
        assert not value['comparison']['live_curves_equal'] and not value['comparison']['result_equal_except_timing']
        cases[record['job']['case']].append(dict(label=record['job']['label'], kind=record['job']['kind'],
                                                 archived=value['comparison']['archived'], replay=replay,
                                                 first_divergent_step=value['comparison']['first_divergent_step'],
                                                 first_divergence_abs=value['comparison']['first_divergence_abs'],
                                                 artifact=record['artifact'], source_artifact=record['job']['artifact']))
    assert all(t['archived']['status'] == 'PASS' for trials in cases.values() for t in trials)
    supported = {name: any(t['replay']['status'] == 'PASS' for t in trials) for name, trials in cases.items()}
    required = [n for n, trials in cases.items() if trials[0]['kind'] == 'required']
    practical = [n for n in cases if n not in required]
    required_passes, practical_passes = sum(supported[n] for n in required), sum(supported[n] for n in practical)
    trial_passes = sum(t['replay']['status'] == 'PASS' for trials in cases.values() for t in trials)
    assert (len(required), len(practical), required_passes, practical_passes, trial_passes) == (9, 10, 8, 8, 39)
    reference = read(REFERENCE)
    cli = []
    for run, build_name, pins in CLI_RUNS:
        folder = ROOT / 'cli' / run
        value = read(folder / 'episodes/linear_skip_d96_beta5__vector_unequal_mass.json.gz')
        run_protocol = read(folder / 'protocol.json.gz')
        assert run_protocol['source_sha256'] == protocol['source_sha256']
        ro, no = reference['result']['observations'], value['result']['observations']
        keys = [k for k, v in ro[0].items() if isinstance(v, float) and k != 'seconds']
        first = next(i for i, (a, b) in enumerate(zip(ro, no)) if any(a[k] != b[k] for k in keys))
        cli.append(dict(run=run, torch=run_protocol['torch'], build=build_name, pinned=pins,
                        exact=untimed(value['result']) == untimed(reference['result']),
                        first_divergent_step=ro[first]['step'],
                        first_divergence_abs=max(abs(ro[first][k] - no[first][k]) for k in keys),
                        replay=observed(value['result'], value['spec']), final={k: no[-1][k] for k in keys}))
    fingerprint = lambda r: json.dumps(untimed(read(ROOT / 'cli' / r / 'episodes/linear_skip_d96_beta5__vector_unequal_mass.json.gz')['result']), sort_keys=True)
    build_independent = all(fingerprint(a) == fingerprint(f'cu126_{a}') for a in ('default', 'aten_avx2', 'all_avx2'))
    driver_matches_cli = (untimed(read(ROOT / 'episodes/linear_skip_d96_beta5__vector_unequal_mass.json.gz')['result'])
                          == untimed(read(ROOT / 'cli/default/episodes/linear_skip_d96_beta5__vector_unequal_mass.json.gz')['result']))
    assert build_independent and driver_matches_cli and not any(c['exact'] for c in cli)
    six = [dict(task=r['task'], **observed(read(ROOT / 'cli/cli_all_six' / r['artifact'])['result'],
                                           read(ROOT / 'cli/cli_all_six' / r['artifact'])['spec']))
           for r in read(ROOT / 'cli/cli_all_six/index.json.gz')['records']]
    report = dict(version='host-replication-v1', claim='rp_logistic_bcap3 9/9 required + 10/10 practical on archived host',
                  host=dict(torch=protocol['torch'], cpu_capability=protocol['cpu_capability'], platform=protocol['platform'],
                            threads=protocol['threads'], processes=protocol['processes']),
                  archived_host=dict(torch='2.13.0+cu126', cpu_capability='AVX2', python='3.12.13'),
                  rule=plan['rule'], required_passes=required_passes, practical_passes=practical_passes,
                  trial_passes=trial_passes, trials=len(index), supported=supported, cases=cases,
                  rare_cli_replays=cli, torch_build_changes_result=not build_independent,
                  driver_equals_official_cli=driver_matches_cli, winner_all_six_cli=six)
    (ROOT / 'leaderboard.json').write_text(json.dumps(report, indent=2) + '\n')
    render(report)
    print(dict(required=f'{required_passes}/9', practical=f'{practical_passes}/10', trials=f'{trial_passes}/{len(index)}'))


def render(report):
    cases, supported = report['cases'], report['supported']
    flipped = [n for n, ok in supported.items() if not ok]
    lines = ['# Second-host replication of the b_cap3 19/19 claim', '',
             f"**Not replicated: {report['required_passes']}/9 required + {report['practical_passes']}/10 practical = "
             f"{report['required_passes'] + report['practical_passes']}/19 on a second CPU host.** "
             'The archived 19/19 evidence rebuilds and verifies unchanged, and it replays exactly on its original host. '
             'This replay keeps identical source hashes, torch 2.13.0, seed 0, specs, policies, architectures and thresholds. '
             f"Every one of the {report['trials']} replayed episodes already differs from its archive at its first measurement "
             '(the rare-mode winner by 3.7e-6 at step 50); differences then amplify. '
             f"{report['trial_passes']}/{report['trials']} archived passing trials still pass. "
             f"Cases without a surviving supporting architecture: **{', '.join(flipped)}**.", '',
             '**Why:** two MKL kernels, narrow output matrix products and float32 sqrt in Adam, round differently by CPU. Seed-0 training amplifies '
             'those one-ULP differences to macroscopic divergence by update ~300. With `MKL_CBWR=AVX2`, a full rare-mode run is bit-identical on native '
             'Intel AVX512 and emulated Intel AVX2 CPUs. That portable mode scores 17/19; the rare mode and ring still fail. '
             'See the [portability analysis](portability/README.md).', '',
             'This is verification of the Codex result, not a new formulation or leaderboard entry. '
             'The formulation leaderboard still counts the archived live evidence. '
             'This host is a robustness audit: only archived passing trials were replayed, so archived failures cannot be promoted by a lucky host.', '',
             '| Case | Archived | This host | Archived passing architectures still passing | Replayed final passing checks |',
             '| --- | --- | --- | ---: | --- |']
    for name, trials in cases.items():
        ok = [t for t in trials if t['replay']['status'] == 'PASS']
        checks = ', '.join(f"{t['label']} {t['replay']['passing_suffix']}" for t in trials)
        lines.append(f"| {name} | PASS | {'PASS' if supported[name] else '**FAIL**'} | {len(ok)}/{len(trials)} | {checks} |")
    lines += ['', 'Required tests list the original host. Passing needs every live metric at each of the final five of 24 checks; EMA is not used.', '',
              '## Archived passing trials that fail here', '',
              '| Case / architecture | Archived final passing checks | Replayed final passing checks | Replayed final failing metrics |',
              '| --- | ---: | ---: | --- |']
    for name, trials in cases.items():
        for t in trials:
            if t['replay']['status'] != 'PASS':
                lines.append(f"| {name} / {t['label']} | {t['archived']['passing_suffix']} | {t['replay']['passing_suffix']} | "
                             f"{', '.join(t['replay']['final_failing']) or 'All final bounds pass; too few final passing checks'} |")
    lines += ['', 'The required eight-mode ring collapses to 5/8 modes with 74.6% good samples. The rare-mode winner ends at component covariance error 2.624 (≤.85) '
              'and minimum spread .092 (≥.15). img_bars4 passes its final four checks but needs five; residual16 is its only archived supporting architecture.', '',
              '## Rare-mode winner: documented command', '',
              '`python -u -m benchmarks.transfer_suite.run_linear_skip --tasks vector_unequal_mass`, unchanged, under two torch 2.13.0 builds and three CPU instruction pins:', '',
              '| Torch build | Pinned instruction set | Exact vs archive | First divergence | Live result | Final passing checks | Final covariance error | Final spread |',
              '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for c in report['rare_cli_replays']:
        lines.append(f"| {c['build']} | {c['pinned']} | {'Yes' if c['exact'] else 'No'} | step {c['first_divergent_step']}, {c['first_divergence_abs']:.1e} | "
                     f"{c['replay']['status']} | {c['replay']['passing_suffix']} | {c['final']['component_covariance_error']:.3f} | {c['final']['component_min_eigen_ratio']:.3f} |")
    lines += ['', 'The two torch builds give bit-identical results on this host, so the archive/host difference is the CPU numerical path, not the wheel. '
              'Pinning MKL/oneDNN to AVX2 changes the trajectory but does not recover the archived one. '
              'The replication driver reproduces the official CLI bit-for-bit on this host, and repeated runs are deterministic. '
              'The CLI plan declared the three CPU-wheel runs; the cu126 runs and the six-toy run were added after those failed, '
              'to isolate the torch build and record the documented full profile.', '',
              "The winner's full six-data CLI profile here is also 3/6, with a different set: "
              + ', '.join(f"{r['task'].replace('vector_', '')} {r['status']} ({r['passing_suffix']})" for r in report['winner_all_six_cli'])
              + '. Archived: rare mass, broad and spiral pass; anisotropic fails.', '',
              '## Interpretation', '',
              'These fixed-seed results are sensitive to floating-point rounding. Cases with several passing architectures or long passing streaks replicate '
              '(broad, spiral, anisotropic, stripes, intensity; blobs via residual16; overlap via d128_l3_f4, d64_l2_f5 and both tanh critics). '
              'Cases resting on one selected architecture do not reliably replicate: the rare 2% mode (1 winner of 58 candidates) and bars4 (residual16 only). '
              'The required eight-mode ring also collapses here. A 19/19 claim should cite host identity alongside seed 0; a host-robust claim '
              'would need passes on more than one CPU host. No threshold, target, seed or scoring rule changed.', '',
              f"Host: torch {report['host']['torch']}, CPU capability {report['host']['cpu_capability']}, {report['host']['platform']}, one thread per episode, "
              f"{report['host']['processes']} concurrent episodes. Archived host: torch 2.13.0+cu126, AVX2.", '',
              '## Reproduction', '', '```bash',
              'python -u -m reports.transfer_suite.host_replication.run --output /tmp/host-replication > /tmp/host-replication.log 2>&1',
              'grep -E "^(DONE|COMPLETE)" /tmp/host-replication.log   # or tail -f',
              'python -m reports.transfer_suite.host_replication.build   # rebuild from archived episodes, no training', '```', '',
              '[Machine-readable results](leaderboard.json) · [Replay log](run.log) · [Declared jobs](plan.json.gz) · [Protocol and source hashes](protocol.json.gz) · '
              '[CLI replay plan](cli/plan.json.gz) · [Archive hashes](archive_manifest.json) · [Tests](tests.log).']
    (ROOT / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    build()
