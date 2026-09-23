"""Rebuild the host-portability explanation from archived artifacts; no training."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

from benchmarks.transfer_suite.protocol import test_verdict
from ..run import untimed

ROOT = Path(__file__).resolve().parent
EPISODE = 'episodes/linear_skip_d96_beta5__vector_unequal_mass.json.gz'


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())


def replication_score(folder):
    cases = defaultdict(list)
    for record in read(folder / 'index.json.gz')['records']:
        cases[(record['job']['case'], record['job']['kind'])].append(record['comparison']['replay']['status'] == 'PASS')
    required = [c for c in cases if c[1] == 'required']
    practical = [c for c in cases if c[1] != 'required']
    return dict(required=sum(any(cases[c]) for c in required), practical=sum(any(cases[c]) for c in practical),
                trials=sum(map(sum, cases.values())), total_trials=sum(map(len, cases.values())),
                unsupported=[c[0] for c in cases if not any(cases[c])])


def build():
    paths = [json.loads(line) for line in (ROOT / 'paths.jsonl').read_text().splitlines()]
    assert not any(p['equals_archive'] for p in paths)
    classes = defaultdict(list)
    for p in paths:
        classes[p['fingerprint']].append(p['label'])
    growth = read(ROOT / 'default_vs_cnr_avx2.json')
    compat = read(ROOT / 'compatible_native_vs_emulated_haswell.json')
    forward = lambda report: {k: v for k, v in report['step1_outputs_and_first_update'].items()
                              if k not in ('d_grad_before_first_update', 'd_after_first_update')}
    assert {k for k, v in forward(growth).items() if v['differing_elements']} == {'g.net.4', 'g.net', 'd.main.net.4', 'd.main.net', 'd.main'}
    assert not any(v['differing_elements'] for v in forward(compat).values())
    assert compat['step1_outputs_and_first_update']['d_grad_before_first_update']['differing_elements'] == 0
    assert compat['step1_outputs_and_first_update']['d_after_first_update']['differing_elements'] > 0
    native = read(ROOT / 'cross_cpu/native_mkl_cnr_avx2' / EPISODE)
    emulated = read(ROOT / 'cross_cpu/emulated_haswell_mkl_cnr_avx2' / EPISODE)
    cross_cpu_exact = untimed(native['result']) == untimed(emulated['result'])
    verdict = test_verdict(native['spec'], native['result'])
    cnr = replication_score(ROOT / 'cnr_avx2_replication')
    default = replication_score(ROOT.parent)
    summary = dict(numerical_paths=paths, fingerprint_classes=classes, archive_fingerprint=paths[0]['archive_fingerprint'],
                   default_vs_cnr_avx2=growth, compatible_native_vs_emulated=compat,
                   cross_cpu_full_run_exact=cross_cpu_exact, cnr_avx2_rare_verdict=verdict['status'],
                   cnr_avx2_rare_passing_suffix=verdict['convergence']['passing_suffix'],
                   scores=dict(default_path=default, mkl_cnr_avx2_path=cnr))
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    render(summary)
    print(dict(paths=len(paths), classes=len(classes), cross_cpu_exact=cross_cpu_exact,
               default=f"{default['required']}/9+{default['practical']}/10", cnr=f"{cnr['required']}/9+{cnr['practical']}/10"))


def render(s):
    g = s['default_vs_cnr_avx2']
    layers = g['step1_outputs_and_first_update']
    fmt = lambda row: f"{row['generator']:.1e} / {row['particles']:.1e} / {row['discriminator']:.1e}"
    lines = ['# Why the archived 19/19 does not replay on other CPUs', '',
             '**The archived rare-mode PASS is one outcome of a chaotic trajectory, pinned to its original CPU numerical path.** '
             'Two MKL kernels round differently on different CPUs. Seed-0 GAN training amplifies those one-ULP differences '
             'until the late checks are effectively independent draws. No fresh run on any tested numerical path passes the rare toy, so 19/19 is not host-portable.', '',
             '## 1. Where the first difference appears', '',
             'Native default path vs MKL CNR AVX2 on the same machine, same process-level seed and code. During the first forward passes only the narrow output layers differ:', '',
             '| Tensor at update 1 | Shape | Differing elements | Max abs difference |', '| --- | --- | ---: | ---: |']
    for name in ('g.net.0', 'g.net.2', 'g.net.4', 'd.main.net.0', 'd.main.net.2', 'd.main.net.4', 'd_grad_before_first_update', 'd_after_first_update'):
        row = layers[name]
        lines.append(f"| {name} | {row['shape']} | {row['differing_elements']} | {row['max_abs_difference']:.2e} |")
    c = s['compatible_native_vs_emulated']['step1_outputs_and_first_update']
    lines += ['', 'Wide hidden layers match bit-for-bit. G 64→2 and D 96→1 differ by about one float32 ULP, so MKL chooses '
              'CPU-specific kernels for these narrow matrix products. Under MKL CNR COMPATIBLE, native AVX512 and emulated Haswell agree on every '
              f"forward output and on all {c['d_grad_before_first_update']['shape'][0]:,} D gradients. The first Adam step still changes "
              f"{c['d_after_first_update']['differing_elements']} D parameters by {c['d_after_first_update']['max_abs_difference']:.1e}.", '',
              'That second source is `torch.sqrt` in the Adam denominator. Its float32 results depend on MKL settings: MKL AVX2 instructions '
              'make every sqrt correctly rounded, while the default AVX512 path misses 73. An initial isolated check found identical lerp, '
              'addcmul and addcdiv results across these CPUs, while sqrt differed:', '',
              '| Numerical path | Step-50 fingerprint | Matches archive | sqrt results not correctly rounded / 10,467 | Adam step digest |',
              '| --- | --- | --- | ---: | --- |']
    for p in s['numerical_paths']:
        lines.append(f"| {p['label']} | `{p['fingerprint']}` | {'yes' if p['equals_archive'] else 'no'} | {p['sqrt_not_correctly_rounded']} | `{p['adam_step_digest']}` |")
    lines += ['', f"The archive's step-50 fingerprint is `{s['archive_fingerprint']}`. There are {len(s['fingerprint_classes'])} distinct fingerprints "
              f"across {len(s['numerical_paths'])} tested paths, and none matches. The archive host reported AVX2; emulated Intel "
              'AVX2 and AMD Zen 2/3 CPUs still miss it. Exact replay therefore needs that machine or its exact MKL kernel path.', '',
              '## 2. How fast the difference grows', '',
              'Relative parameter difference, G / particles / D, between the two native paths:', '',
              '| G update | Relative difference |', '| ---: | --- |']
    lines += [f"| {row['g_update']} | {fmt(row)} |" for row in g['relative_parameter_difference']]
    lines += ['', 'Adam uses beta1=0, so the first update is nearly sign-like. Parameters with tiny gradients turn ULP gradient changes into '
              'parameter changes of about 1e-6. The GAN then grows the difference by roughly e every 20 updates until it saturates near '
              'update 300 of 1,200. The five scored checks at updates 1,000–1,200 are no longer tied to the archived trajectory. '
              'At this rate float64 would delay saturation by only several hundred updates. That estimate is extrapolated, not run.', '',
              '## 3. What makes fresh runs portable', '',
              f"With `MKL_CBWR=AVX2`, the full 1,200-update rare-mode run is **{'bit-identical' if s['cross_cpu_full_run_exact'] else 'NOT identical'}** "
              'between native Intel AVX512 and an emulated Intel Haswell (AVX2) CPU, excluding timing. That run is '
              f"**{s['cnr_avx2_rare_verdict']}** with {s['cnr_avx2_rare_passing_suffix']} final passing checks. "
              'The same setting does not cover AMD: emulated Zen 2 gives another fingerprint. CNR COMPATIBLE unifies emulated Intel and AMD but '
              'not native AVX512, because the MKL vector sqrt still differs. A cross-vendor exact mode needs code-level control of these two kernels, not only environment variables.', '',
              '| Fresh numerical path | Required | Practical | Total | Passing archived trials | Cases without support |',
              '| --- | ---: | ---: | ---: | ---: | --- |', '| Archived host | 9/9 | 10/10 | 19/19 | 46/46 | — |']
    for label, key in (('This host, default', 'default_path'), ('This host, MKL CNR AVX2 (Intel-portable)', 'mkl_cnr_avx2_path')):
        r = s['scores'][key]
        lines.append(f"| {label} | {r['required']}/9 | {r['practical']}/10 | {r['required'] + r['practical']}/19 | {r['trials']}/{r['total_trials']} | {', '.join(r['unsupported'])} |")
    lines += ['', 'The required eight-mode ring and the rare 2% mode fail on both fresh paths. bars4 passes under CNR AVX2 but not default. '
              'Every other case retains a passing architecture on all three paths. The passing architecture can change; for example, '
              'both anisotropic D128 variants fail under CNR AVX2 while the original D passes.', '',
              '## Recommendations', '',
              '- Report the numerical path with seed 0. For future evidence, run with `MKL_CBWR=AVX2` so any Intel AVX2/AVX512 host reproduces it exactly.',
              '- Do not call a cell host-portable unless it passes on at least two numerical paths, for example default and CNR AVX2. The rare mode and ring currently fail that test.',
              '- For cross-vendor exact replay, make Adam\'s sqrt correctly rounded and give narrow output layers a fixed-order reduction in a declared evaluation mode, then rerun all 19 cases. This changes archived numerics, so it would require new evidence.',
              '', '## Reproduce', '', '```bash',
              'python3 -m reports.transfer_suite.host_replication.portability.paths /tmp/paths     # needs qemu-user for emulated CPUs',
              'python3 -m reports.transfer_suite.host_replication.portability.trace /tmp/a.pt',
              'MKL_CBWR=AVX2 python3 -m reports.transfer_suite.host_replication.portability.trace /tmp/b.pt',
              'python3 -m reports.transfer_suite.host_replication.portability.analyze /tmp/a.pt /tmp/b.pt /tmp/growth.json',
              'MKL_CBWR=AVX2 python -u -m reports.transfer_suite.host_replication.run --output /tmp/cnr-avx2',
              'python3 -m reports.transfer_suite.host_replication.portability.build', '```', '',
              '[Summary](summary.json) · [Numerical paths](paths.jsonl) · [Growth and localization](default_vs_cnr_avx2.json) · '
              '[COMPATIBLE native vs emulated](compatible_native_vs_emulated_haswell.json) · [Cross-CPU full runs](cross_cpu/) · '
              '[CNR AVX2 replication](cnr_avx2_replication/run.log).']
    (ROOT / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    build()
